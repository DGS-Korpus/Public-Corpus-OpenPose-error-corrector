import argparse
import json
import os
import re
import sys
from itertools import combinations, chain

import numpy as np

assert sys.version_info >= (3, 7), 'Requires Python 3.7 or greater to ensure that dict key order is insertion order.'

# Regular expression used to remove JSON newlines and indentation from within keypoint models.
KEYPOINTS_KEY_VALUE_RE = re.compile(r'("[\w_]+?_keypoints_\dd": \[)([\n\d., -e]+?)(\])', flags=re.DOTALL)

# In the numpy array representation, the keypoints of all four OpenPose models are concatenated on a single dimension.
# This dict specifies the index range of each model on that dimension.
MODEL_INDEX_RANGE = {'pose_keypoints_2d': (0, 25), 'face_keypoints_2d': (25, 95),
                     'hand_left_keypoints_2d': (95, 116), 'hand_right_keypoints_2d': (116, 137)}

# Relevance weights for keypoints indicating how important a keypoint is for the centre of mass of a person.
POSE_KEYPOINT_RELEVANCE = [15,  # 0  (Nose)
                           20,  # 1  (Collarbone)
                           15,  # 2  (Right shoulder)
                           5,  # 3  (Right elbow)
                           3,  # 4  (Right wrist)
                           15,  # 5  (Left shoulder)
                           5,  # 6  (Left elbow)
                           3,  # 7  (Left wrist)
                           20,  # 8  (Center hip)
                           18,  # 9  (Right hip)
                           3,  # 10 (Right knee)
                           1,  # 11 (Right ankle)
                           15,  # 12 (Left hip)
                           3,  # 13 (Left knee)
                           1,  # 14 (Left ankle)
                           10,  # 15 (Right eye)
                           10,  # 16 (Left eye)
                           10,  # 17 (Right ear)
                           10,  # 18 (Left ear)
                           1,  # 19 (Left ball of foot)
                           1,  # 20 (Left toes)
                           1,  # 21 (Left heel)
                           1,  # 22 (Right ball of foot)
                           1,  # 23 (Right toes)
                           1,  # 24 (Right heel)
                           ]

# As participants in the Public DGS Corpus recordings sit in a pre-determined area of the recording for most of the
# time, we can assign roles based on where in the image a person is seen:
# Informant B sits on the left, the moderator sits in the centre and informant A sits on the right.
ROLE2POSITION_RANGE = {'b': (0, .33),
                       'm': (.33, .66),
                       'a': (.66, 1),
                       }


class Frame:
    """
    A convenience class to collect all relevant information for specifc single frame in one place.
    """

    def __init__(self, index, input_array):
        self.index = int(index)
        self.input_array = input_array
        self.output_array = input_array
        self.previous_frame = None
        self.output_index2input_indices = {i: [i] for i in range(len(input_array))}
        self.output_index2prev_output_index = {}
        self.output_index2role = {}
        self.role2output_index = {}


# Load/write functions
def load_openpose_json(filename):
    with open(filename) as f:
        return json.load(f)


def _ensure_dir(filename):
    """
    Check whether the directory for the given filename exists and if it does not, create it.
    """
    filepath = os.path.dirname(filename)
    os.makedirs(filepath, exist_ok=True)


def _deindent_key_value(match):
    """
    To be called by the regular expression substitution that de-indents OpenPose arrays.
    Given a fully indented json file, the substitution command should be:
    json_output, subs = KEYPOINTS_KEY_VALUE_RE.sub(_deindent_key_value, json_indented)
    """
    key_string = match.group(1)
    array_content_string = match.group(2)
    array_close_string = match.group(3)
    deindented_content_items = [item.strip() for item in array_content_string.split('\n')]
    deindented_content_string = ' '.join(deindented_content_items).strip()
    return '{}{}{}'.format(key_string, deindented_content_string, array_close_string)


def write_openpose_json(filename, openpose_data):
    _ensure_dir(filename)
    op_data_fullindent_json = json.dumps(openpose_data, sort_keys=False, indent=2)
    op_data_slimindent_json = KEYPOINTS_KEY_VALUE_RE.sub(_deindent_key_value, op_data_fullindent_json)

    with open(filename, 'w') as w:
        w.write(op_data_slimindent_json)


# Data conversion
def get_frame2array_mapping(openpose_data_for_perspective):
    """
    Convert OpenPose the person models of each frame from a dict/list structure to a numpy array.
    The array is of shape (n, 137, 3):
    - The first dimension represents the n people in the frame.
    - The second dimension represents all 137 keypoints of a person, starting with the 25 pose model keypoints,
      followed by 70 face, 21 left hand and 21 right hand model keypoints.
    - The third dimension represents the three components of each keypoint: x-coordinate, y-coordinate and confidence.
    :param openpose_data_for_perspective: The dict/list structure of a transcript trakt.
    :return: A dict mapping (int->np.array) where the key is a frame index and the value a people-keypoint array.
    """
    model_names = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']

    openpose_frames = openpose_data_for_perspective['frames']
    frame2array = {}
    for frame_index, frame in openpose_frames.items():
        openpose_people = frame['people']
        person_arrays = []
        # Create an array for each person
        for openpose_person in openpose_people:
            # Select models in the correct order
            models = [openpose_person[m] for m in model_names]
            # Flatten list of models and turn into array
            person_flat_array = np.fromiter(chain.from_iterable(models), float)
            # Turn the flat person array into one where each keypoint is represented as an array of length 3
            person_keypoint_array = person_flat_array.reshape((-1, 3))
            person_arrays.append(person_keypoint_array)
        # Combine all single person arrays into a single array of all people in the given frame
        people_array = np.array(person_arrays)
        frame2array[int(frame_index)] = people_array
    return frame2array


# Computing the centre point of a person
def get_mean_person_position(person_array):
    available_keypoints_array = get_available_person_pose_keypoints(person_array)
    avg_xpos, avg_ypos = available_keypoints_array.mean(axis=0)
    return avg_xpos, avg_ypos


def get_average_person_position(person_array):
    """
    Determine the position of a person, based on a weighted average that gives preference to the body's center of mass.
    """
    # Determine keypoints available for this person
    pose_position_array, availability_array = _get_available_person_pose_keypoints(person_array)
    # Set keypoints weights (and set unavailable keypoints to weight of 0)
    keypoint_weights = [w if availability_array[k] else 0 for k, w in enumerate(POSE_KEYPOINT_RELEVANCE)]
    # Compute weighted average position
    avg_xpos, avg_ypos = np.average(pose_position_array, axis=0, weights=keypoint_weights)
    return avg_xpos, avg_ypos


# Determine which single-view cameras were involved
def determine_public_roles(transcript_data, publish_moderator):
    public_roles = set()
    for track_data in transcript_data:
        camera = track_data['camera']
        if camera == 'a1':
            public_roles.add('a')
        elif camera == 'b1':
            public_roles.add('b')
        elif camera == 'c':
            pass
        else:
            raise ValueError("Unexpected camera: {}".format(camera))

    if publish_moderator:
        public_roles.add('m')

    # Enforce role order
    role_order = {role: index for index, role in enumerate('bma')}
    ordered_public_roles = sorted(public_roles, key=role_order.get)

    return ordered_public_roles


# Step 1: Repairing and ordering OpenPose people.
def _get_available_person_pose_keypoints(person_array):
    pose_start, pose_end = MODEL_INDEX_RANGE['pose_keypoints_2d']
    pose_position_array = person_array[pose_start:pose_end, :2]
    return pose_position_array, np.all(pose_position_array != 0, axis=1)


def get_available_person_pose_keypoints(person_array):
    pose_position_array, person_pose_keypoint_availability = _get_available_person_pose_keypoints(person_array)
    return pose_position_array[person_pose_keypoint_availability]


def get_standard_deviation_of_person(person_array):
    available_keypoints_array = get_available_person_pose_keypoints(person_array)
    std_xpos, std_ypos = available_keypoints_array.std(axis=0)
    return std_xpos, std_ypos


def merge_people_fragments(frame_array, mergeable_people_combos):
    combo2merged_array = {}
    for mergeable_people_combo in mergeable_people_combos:
        merged_person_array = np.sum(frame_array[mergeable_people_combo, ], axis=0)

        std_xpos, std_ypos = get_standard_deviation_of_person(merged_person_array)
        if std_xpos < 200:
            combo2merged_array[mergeable_people_combo] = merged_person_array

    return combo2merged_array


def find_mergeable_person_fragments(frame_array):
    """
    Determine which person entries could possibly be merged (i.e. are complementary)
    """
    frame_array_intbool = np.all(frame_array.astype(bool).astype(int), axis=2)
    mergeable_entry_combos = []
    for r in range(len(frame_array_intbool), 1, -1):
        for combo in combinations(range(len(frame_array_intbool)), r):
            # Sum of 2-d array shows us that this merge would give us:
            # - missing keypoint == 0
            # - keypoint available in exactly one fragment == 1
            # - keypoint available in multiple fragments >= 2 (i.e. an illegal overlap)
            keypoint_overlap = np.sum(frame_array_intbool[combo, ], axis=0)
            if np.all(keypoint_overlap < 2):  # No overlapping keypoints between fragments
                mergeable_entry_combos.append(combo)

    return mergeable_entry_combos


def reduce_subsets(combo2merged_array):
    """
    Given a set of possible fragment sets that can be merged into a proper person, if one of the sets is a superset to
    all the other sets, reduce the mapping to this single possibility. Otherwise give up (for now) and just return all
    possible mappings again.
    Example 1: {(1,2,3):..., (1,2):..., (1,3):..., (2,3):...} -> {(1,2,3):...}
    Example 1: {(1,2):..., (3,4):...} -> {(1,2):..., (3,4):...}
    :param combo2merged_array: Mapping from sets of (fragmented person) indices to the array of the restored person.
    :return: A copy of the input mapping, possibly reduced to the one fragment combination that covers all fragments.
    """
    combo2merged_array = combo2merged_array.copy()
    if len(combo2merged_array) <= 1:
        return combo2merged_array

    combo_sets = [set(combo) for combo in combo2merged_array]
    for super_combo, merged_array in combo2merged_array.items():
        super_combo_set = set(super_combo)
        is_superset = all(super_combo_set.issuperset(combo) for combo in combo_sets)

        if is_superset:
            return {super_combo: merged_array}

    return combo2merged_array


def get_defragmented_people_array(frame_array, combo2merged_array):
    """
    Create a new frame array that contains all people after fragmentation has been repaired.
    :param frame_array: The original frame array with dimensions [people, keypoints, x/y/conf]
    :param combo2merged_array: Mapping from sets of (fragmented person) indices to the array of the restored person.
    :return: A new frame array containing the restored people and two mappings of indices from input array to
             output array and vice versa.
    """
    unassigned_people = set(range(len(frame_array)))
    person2combo_array_tuple = {}
    # Assigned merged arrays to a single person index
    for combo, merged_array in combo2merged_array.items():
        combo = tuple(sorted(combo))
        person2combo_array_tuple[combo[0]] = (combo, merged_array)
        for original_index in combo:
            unassigned_people.discard(original_index)

    # Assign regular people
    for original_index in unassigned_people:
        single_person_combo = (original_index,)
        person2combo_array_tuple[original_index] = (single_person_combo, frame_array[original_index])

    person_arrays = []
    original_index2defragmented_index = {}
    defragmented_index2original_indices = {}
    combo_array_tuples = [combo_array_tuple for i, combo_array_tuple in sorted(person2combo_array_tuple.items())]
    for defrag_index, (combo, person_array) in enumerate(combo_array_tuples):
        person_arrays.append(person_array)
        defragmented_index2original_indices[defrag_index] = combo
        for original_index in combo:
            original_index2defragmented_index[original_index] = defrag_index

    # Resort mapping for iteration convenience
    defragmented_index2original_indices = {k: vs for k, vs in sorted(defragmented_index2original_indices.items())}

    defragmented_frame_array = np.array(person_arrays)
    return defragmented_frame_array, original_index2defragmented_index, defragmented_index2original_indices


def get_sorted_person_order(frame_array, position_func=get_mean_person_position):
    """
    Determine order of people based on their average position.
    What is meant by "average positon" depends on the behaviour of position_func.
    :param frame_array: The frame array containing all people arrays of a given frame.
    :param position_func: A function that takes a person array and returns a (x,y) coordinate tuple.
    :return: A list containing the person indices in their sorted order.
    """

    position_index_tuples = []
    for person_index, this_person_array in enumerate(frame_array):
        avg_xpos, avg_ypos = position_func(this_person_array)
        position_index_tuples.append((avg_xpos, avg_ypos, person_index))
    position_index_tuples.sort()
    person_order = [index for avg_xpos, avg_ypos, index in position_index_tuples]
    return tuple(person_order)


def reorder_output_array(frame_array, input_index2output_index, output_index2input_indices, new_output_index_order):
    reordered_frame_array = frame_array[new_output_index_order, ]
    output2new = {output: new for new, output in enumerate(new_output_index_order)}
    input2new = {input_i: output2new[output] for input_i, output in input_index2output_index.items()}
    new2inputs = {output2new[output]: inputs for output, inputs in output_index2input_indices.items()}
    new2inputs = {k: vs for k, vs in sorted(new2inputs.items())}
    return reordered_frame_array, input2new, new2inputs


def get_pose_keypoints_diff2people(array1, array2):
    # Extract a subarray that only contains the pose model and only covers the x- and y coordinate, not the confidence.
    pose_start, pose_end = MODEL_INDEX_RANGE['pose_keypoints_2d']
    pose_keypoints_array1 = array1[:, pose_start:pose_end, :2]
    pose_keypoints_array2 = array2[:, pose_start:pose_end, :2]
    # Determine which keypoints are available for both arrays
    overlap = np.all((pose_keypoints_array1 != 0) & (pose_keypoints_array2 != 0), axis=2)
    has_overlap = np.any(overlap)
    if has_overlap:
        # Compute the positional difference between the arrays
        diff_array = pose_keypoints_array1[overlap] - pose_keypoints_array2[overlap]
        return diff_array
    else:  # There is no overlap
        return None


def get_pose_keypoints_diff2person(array1, array2):
    # Extract a subarray that only contains the pose model and only covers the x- and y coordinate, not the confidence.
    pose_start, pose_end = MODEL_INDEX_RANGE['pose_keypoints_2d']
    pose_keypoints_array1 = array1[pose_start:pose_end, :2]
    pose_keypoints_array2 = array2[pose_start:pose_end, :2]
    # Determine which keypoints are available for both arrays
    overlap = np.all((pose_keypoints_array1 != 0) & (pose_keypoints_array2 != 0), axis=1)
    has_overlap = np.any(overlap)
    if has_overlap:
        # Compute the positional difference between the arrays
        diff_array = pose_keypoints_array1[overlap] - pose_keypoints_array2[overlap]
        return diff_array
    else:  # There is no overlap
        return None


def merge_overlapping_people_fragments(frame_array, prev_frame_array, overlapping_person_indices,
                                       linked_prev_person_index):
    limb_connector_keypoints = {1, 2, 5, 8, 9, 12}
    frame_array_bool = np.all(frame_array.astype(bool).astype(int), axis=2)
    # Compute overlap between keypoints of person fragments. A keypoint value of >=2 indicates an overlap.
    overlap = np.sum(frame_array_bool[overlapping_person_indices, ], axis=0)
    overlapping_keypoints = {kp for kp, matches in enumerate(overlap) if matches >= 2}
    # For each overlapping keypoint determine which fragment's keypoint is closest to the one from the previous frame
    keypoint2preferred_person = {}
    if overlapping_keypoints.issubset(limb_connector_keypoints):
        linked_prev_person_array = prev_frame_array[linked_prev_person_index]
        for kp in overlapping_keypoints:
            instances_of_keypoint_array = frame_array[overlapping_person_indices, kp, :2]
            linked_prev_keypoint_array = linked_prev_person_array[kp, :2]
            keypoint2prev_diff = instances_of_keypoint_array - linked_prev_keypoint_array
            keypoint2prev_distance = np.linalg.norm(keypoint2prev_diff, axis=1)
            smallest_distance_index = np.argmin(keypoint2prev_distance)
            preferred_person_index = overlapping_person_indices[smallest_distance_index]
            keypoint2preferred_person[kp] = preferred_person_index

        # Merge people fragments and clean up overlapping keypoints.
        merged_person_array = np.sum(frame_array[overlapping_person_indices, ], axis=0)
        for keypoint, preferred_person in keypoint2preferred_person.items():
            merged_person_array[keypoint] = frame_array[preferred_person][keypoint]
    else:
        merged_person_array = None

    return merged_person_array, overlapping_keypoints


def repair_person_order(frame2array, max_std_basic=50, max_std_exhaustive=50):
    id2frame = {}

    prev_f = -1
    for i, (f, input_frame_array) in enumerate(frame2array.items()):
        id2frame[f] = frame = Frame(f, input_frame_array)
        frame.previous_frame = prev_f

        # If there are fragmented people, repair them
        mergeable_people_combos = find_mergeable_person_fragments(input_frame_array)
        fragment_combo2merged_array = merge_people_fragments(input_frame_array, mergeable_people_combos)
        fragment_combo2merged_array = reduce_subsets(fragment_combo2merged_array)
        frame.output_array, input_index2output_index, frame.output_index2input_indices = get_defragmented_people_array(
            input_frame_array, fragment_combo2merged_array)

        # Reorder people from left to right
        sorted_person_order = get_sorted_person_order(frame.output_array, position_func=get_average_person_position)
        frame.output_array, input_index2output_index, frame.output_index2input_indices = reorder_output_array(
            frame.output_array, input_index2output_index, frame.output_index2input_indices, sorted_person_order)

        # Frame Linking: Determine who people from this frame were in the previous frame
        prev_frame = id2frame.get(prev_f)
        if prev_frame is None:  # First frame
            pass
        else:
            basic_approach_succeeded = False

            # Basic approach: Assume person order is already correct and the same number of people are in both frames.
            if len(frame.output_array) == len(prev_frame.output_array):
                output_index2prev_output_index = {}
                diff_array = get_pose_keypoints_diff2people(frame.output_array, prev_frame.output_array)
                if diff_array is not None:
                    standard_dev = diff_array.std()
                    if standard_dev < max_std_basic:
                        basic_approach_succeeded = True
                        for output_index in range(len(frame.output_array)):
                            output_index2prev_output_index[output_index] = output_index
                frame.output_index2prev_output_index = output_index2prev_output_index

            # Exhaustive approach: Try all person pairings between the two frames.
            if not basic_approach_succeeded:
                output_index2prev_output_index = {}
                prev_output_index2output_indices = {}
                for output_index, output_person_array in enumerate(frame.output_array):
                    # Compute the difference of this person to all people from the previous frame
                    std_previndex_tuples = []
                    for previndex, prev_output_person_array in enumerate(prev_frame.output_array):
                        diff_array = get_pose_keypoints_diff2person(output_person_array,
                                                                    prev_output_person_array)
                        if diff_array is not None:  # Only consider if the persons have any overlapping keypoints
                            std_previndex_tuples.append((diff_array.std(), previndex))

                    # Select the person from the previous frame with the smallest difference to this frame's person.
                    std_previndex_tuples.sort()
                    best_std, best_previndex = std_previndex_tuples[0]

                    # Check whether the match is realistic
                    if best_std < max_std_exhaustive:
                        output_index2prev_output_index[output_index] = best_previndex
                        prev_output_index2output_indices.setdefault(best_previndex, []).append(output_index)
                    else:
                        output_index2prev_output_index[output_index] = None
                        prev_output_index2output_indices.setdefault(None, []).append(output_index)
                frame.output_index2prev_output_index = output_index2prev_output_index

                # Check whether more than one person was linked to the prior frame and try to repair such fragmentation.
                overlap_combo2merged_array = {}
                for prev_output_index, overlapping_output_indices in prev_output_index2output_indices.items():
                    overlapping_output_indices = tuple(overlapping_output_indices)
                    if prev_output_index is not None and len(overlapping_output_indices) >= 2:
                        overlapped_person_array, overlapping_keypoints = merge_overlapping_people_fragments(
                            frame.output_array, prev_frame.output_array, overlapping_output_indices, prev_output_index)
                        if overlapped_person_array is None:
                            # Merging the overlapping people failed.
                            # Pick the one with the most keypoints as the actual person
                            # All others are dropped for the purpose of linking.
                            output_index_with_most_kp = None
                            most_upper_kp = -1
                            most_full_kp = -1
                            for overlapping_output_index in overlapping_output_indices:
                                _, keypoint_available_array = _get_available_person_pose_keypoints(
                                    frame.output_array[overlapping_output_index])
                                upper_keypoint_count4person = np.sum(keypoint_available_array.astype(int))
                                full_keypoint_count4person = np.sum(keypoint_available_array.astype(int))
                                if (upper_keypoint_count4person > most_upper_kp
                                        or (upper_keypoint_count4person == most_upper_kp
                                            and full_keypoint_count4person > most_full_kp)):
                                    output_index_with_most_kp = overlapping_output_index
                                    most_upper_kp = upper_keypoint_count4person
                                    most_full_kp = full_keypoint_count4person

                            for overlapping_output_index in overlapping_output_indices:
                                if overlapping_output_index != output_index_with_most_kp:
                                    frame.output_index2prev_output_index[overlapping_output_index] = None
                            prev_output_index2output_indices[prev_output_index] = [output_index_with_most_kp]
                        else:
                            overlap_combo2merged_array[overlapping_output_indices] = overlapped_person_array

                # If overlapping fragments were succesfully merged, integrate them into the repaired array.
                if overlap_combo2merged_array:
                    overlap_frame_array, output_i2overlap_i, overlap_i2outout_is = get_defragmented_people_array(
                        frame.output_array, overlap_combo2merged_array)

                    # Clean index mappings
                    input_index2overlap_index = {inp: output_i2overlap_i[outp] for inp, outp in
                                                 input_index2output_index.items()}
                    overlap_index2input_indices = {}
                    for input_index, overlap_index in input_index2overlap_index.items():
                        overlap_index2input_indices.setdefault(overlap_index, []).append(input_index)
                    overlap_index2input_indices = {k: tuple(vs) for k, vs in
                                                   sorted(overlap_index2input_indices.items())}

                    overlap_index2prev_output_index = {ov: frame.output_index2prev_output_index[outs[0]] for ov, outs in
                                                       overlap_i2outout_is.items()}

                    # Overwrite original output files with overlap-repair output files
                    frame.output_array = overlap_frame_array
                    frame.output_index2input_indices = overlap_index2input_indices
                    frame.output_index2prev_output_index = overlap_index2prev_output_index

        # Store information for access during later frames
        prev_f = f

    return id2frame


# Step 2: Assign roles (informant A, informant B, moderator) to persons
def set_roles(id2frame, pixel_width):
    # Determine how many potential identities exist throughout the frames
    identity2person_data_tuple = {}
    frame2identity2person = {}
    frame2person2identity = {}

    next_identity = 0
    prev_index2identity = {}
    for frame_id, frame in id2frame.items():
        frame.output_index2role = {}
        this_index2identity = {}
        for output_index in range(len(frame.output_array)):
            prev_out_index = frame.output_index2prev_output_index.get(output_index)
            if prev_out_index is None:
                identity = str(next_identity)
                next_identity += 1
            else:
                identity = prev_index2identity.get(prev_out_index)

            position = get_average_person_position(frame.output_array[output_index])

            person_data_tuple = (frame_id, output_index, position)
            identity2person_data_tuple.setdefault(identity, []).append(person_data_tuple)
            frame2identity2person.setdefault(frame_id, {})[identity] = output_index
            frame2person2identity.setdefault(frame_id, {})[output_index] = position

            this_index2identity[output_index] = identity
            frame.output_index2role[output_index] = identity
        prev_index2identity = this_index2identity

    # Group identities by the role they fulfill (based on where in the picture they are most of the time)
    role2identity2person_data_tuples = {}
    for identity, person_data_tuples in identity2person_data_tuple.items():
        avg_positions = [t[2] for t in person_data_tuples]
        mean_x_position, mean_y_position = np.median(np.array(avg_positions, ), axis=0)
        perc_x_position = mean_x_position / pixel_width
        role = None
        for role, (min_x, max_x) in ROLE2POSITION_RANGE.items():
            if min_x < perc_x_position < max_x:
                break
        role2identity2person_data_tuples.setdefault(role, {})[identity] = person_data_tuples

    # Select which identity is properly fulfilling its role.
    role2person_data_tuples = {}
    for role, identity2person_data_tuples in sorted(role2identity2person_data_tuples.items()):
        role2person_data_tuples[role] = role_data_tuples = []
        frames_in_role = set()
        combined_identities = []
        for identity, person_data_tuples in sorted(identity2person_data_tuples.items(),
                                                   key=lambda x: len(x[1]), reverse=True):
            frames_in_identity = {t[0] for t in person_data_tuples}

            if frames_in_role.isdisjoint(frames_in_identity):  # The two identities can both represent this role
                frames_in_role.update(frames_in_identity)
                role_data_tuples.extend(person_data_tuples)
                combined_identities.append(identity)
            else:  # Two identities of the same role are overlapping in some frames
                # If the role overlap occurs at the start/end of the identities, it might be a fragmentation issue.
                unproblematic_frames_of_identity = frames_in_identity - frames_in_role

                for person_data_tuple in person_data_tuples:
                    frame_index = person_data_tuple[0]
                    combined_some_frames = False
                    if frame_index in unproblematic_frames_of_identity:
                        frames_in_role.add(frame_index)
                        role_data_tuples.append(person_data_tuple)
                        combined_some_frames = True
                    else:
                        # This identity does not fit with the more frequent one. It gets a backup role instead.
                        backup_role = role + identity
                        role2person_data_tuples.setdefault(backup_role, []).append(person_data_tuple)

                    if combined_some_frames:
                        combined_identities.append(identity)

    # Apply role information to frame object
    for role, person_data_tuples in sorted(role2person_data_tuples.items()):
        for frame_id, output_index, position in person_data_tuples:
            frame = id2frame[frame_id]
            frame.output_index2role[output_index] = role
            frame.role2output_index[role] = output_index


# Step 3: Apply the repairs to the dict/list structure of the JSON format
def get_person_dict():
    """
    Returns a new person dict where the model keypoint lists are empty.
    """
    keypoint_locations = ['pose_keypoints_2d', 'face_keypoints_2d',
                          'hand_left_keypoints_2d', 'hand_right_keypoints_2d',
                          'pose_keypoints_3d', 'face_keypoints_3d',
                          'hand_left_keypoints_3d', 'hand_right_keypoints_3d']
    op_person = {location: [] for location in keypoint_locations}
    return op_person


def get_zero_filled_person_dict():
    """
    Returns a new person dict where the model keypoint lists are filled with zeros.
    """
    person_dict = get_person_dict()
    person_dict['pose_keypoints_2d'] = [0 for _ in range(75)]
    person_dict['face_keypoints_2d'] = [0 for _ in range(210)]
    person_dict['hand_left_keypoints_2d'] = [0 for _ in range(63)]
    person_dict['hand_right_keypoints_2d'] = [0 for _ in range(63)]
    return person_dict


def create_person_dict_from_array(person_array):
    person_dict = get_person_dict()
    for model, (start, end) in MODEL_INDEX_RANGE.items():
        model_array = person_array[start:end]
        flat_model_array = model_array.flatten()
        flat_model_list = flat_model_array.tolist()
        person_dict[model] = flat_model_list
    return person_dict


def create_person_dict_from_frame_obj(public_role, frame, input_people_dict):
    output_index = frame.role2output_index.get(public_role)
    if output_index is None:
        output_person_dict = get_zero_filled_person_dict()
    else:
        input_indices = frame.output_index2input_indices.get(output_index, tuple())
        if len(input_indices) == 0:
            output_person_dict = get_zero_filled_person_dict()
        elif len(input_indices) == 1:
            input_index = input_indices[0]
            output_person_dict = input_people_dict[input_index]
        else:
            person_array = frame.output_array[output_index]
            output_person_dict = create_person_dict_from_array(person_array)
    return output_person_dict


def apply_repairs_to_perpective_dict(perspective_dict, id2frames, public_roles, include_fragments=True):
    proper_roles = {'a', 'm', 'b'}

    input_frames_dict = perspective_dict['frames']
    output_frames_dict = {}
    for frame_index, frame in id2frames.items():
        frame_str_index = str(frame_index)
        frame_dict = input_frames_dict[frame_str_index]
        input_people_dict = frame_dict['people']

        output_people_dict = []
        output_frame_dict = {'version': frame_dict['version'], 'people': output_people_dict}
        output_frames_dict[frame_str_index] = output_frame_dict

        for public_role in public_roles:
            output_person_dict = create_person_dict_from_frame_obj(public_role, frame, input_people_dict)
            output_people_dict.append(output_person_dict)

        if include_fragments:
            for role in frame.role2output_index:
                if role not in proper_roles:
                    output_person_dict = create_person_dict_from_frame_obj(role, frame, input_people_dict)
                    output_people_dict.append(output_person_dict)

    output_perspective_dict = {'id': perspective_dict['id'], 'camera': perspective_dict['camera'],
                               'width': perspective_dict['width'], 'height': perspective_dict['height'],
                               'frames': output_frames_dict}

    return output_perspective_dict


# Main function
def repair_openpose_frames(transcript_data, publish_moderator):
    public_roles = determine_public_roles(transcript_data, publish_moderator)

    repaired_transcript_data = []
    for r, recording_data in enumerate(transcript_data):
        # Repairs, reordering and role filtering are only applied to the total perspective
        if recording_data['camera'] == 'c':
            frame_array = get_frame2array_mapping(recording_data)
            index2frame = repair_person_order(frame2array=frame_array)
            set_roles(index2frame, pixel_width=recording_data['width'])
            repaired_recording_data = apply_repairs_to_perpective_dict(
                recording_data, index2frame, public_roles=public_roles, include_fragments=False)
        else:
            repaired_recording_data = recording_data

        repaired_transcript_data.append(repaired_recording_data)

    return repaired_transcript_data


def main():
    parser = argparse.ArgumentParser(
        description='Repair OpenPose fragmentation and ghost errors and create persistent person order.')
    parser.add_argument('INPUT', help='JSON file structured in the Public DGS Corpus OpenPose wrapper format '
                                      '(see https://doi.org/10.25592/uhhfdm.842).')
    parser.add_argument('OUTPUT', help='Filename for the corrected JSON file.')
    parser.add_argument('--publishmoderator', '--moderator', '--mod', '-m', action='store_true',
                        help='Also include moderator data in the output.')
    args = parser.parse_args()

    input_data = load_openpose_json(filename=args.INPUT)
    repaired_data = repair_openpose_frames(transcript_data=input_data, publish_moderator=args.publishmoderator)
    write_openpose_json(filename=args.OUTPUT, openpose_data=repaired_data)


if __name__ == '__main__':
    main()

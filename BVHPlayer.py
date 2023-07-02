from viewer import SimpleViewer
import numpy as np
from scipy.spatial.transform import Rotation as R

class HierarchyParser(object):

    def __init__(self, bvh_file_path):
        self.lines = self.get_hierarchy_lines(bvh_file_path)
        self.line_number = 0

        self.root_position_channels = []
        self.joint_rotation_channels = []

        self.joint_names = []
        self.joint_parents = []
        self.joint_offsets = []

    def get_hierarchy_lines(self, bvh_file_path):
        hierarchy_lines = []
        for line in open(bvh_file_path, 'r'):
            line = line.strip()
            if line.startswith('MOTION'):
                break
            else:
                hierarchy_lines.append(line)

        return hierarchy_lines
    
    def parse_offset(self, line):
        return [float(x) for x in line.split()[1:]]
    
    def parse_channels(self, line):
        return [x for x in line.split()[2:]]

    def parse_root(self, parent=-1):
        self.joint_parents.append(parent)

        self.joint_names.append(self.lines[self.line_number].split()[1])
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find root offset')
        self.line_number += 1

        if self.lines[self.line_number].startswith('CHANNELS'):
            channels = self.parse_channels(self.lines[self.line_number])
            if self.lines[self.line_number].split()[1] == '3':
                self.joint_rotation_channels.append((channels[0], channels[1], channels[2]))
            elif self.lines[self.line_number].split()[1] == '6':
                self.root_position_channels.append((channels[0], channels[1], channels[2]))
                self.joint_rotation_channels.append((channels[3], channels[4], channels[5]))
        else:
            print('cannot find root channels')
        self.line_number += 1

        while self.lines[self.line_number].startswith('JOINT'):
            self.parse_joint(0)
        self.line_number += 1

    def parse_joint(self, parent):
        self.joint_parents.append(parent)

        index = len(self.joint_names)
        self.joint_names.append(self.lines[self.line_number].split()[1])
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find joint offset')
        self.line_number += 1

        if self.lines[self.line_number].startswith('CHANNELS'):
            channels = self.parse_channels(self.lines[self.line_number])
            if self.lines[self.line_number].split()[1] == '3':
                self.joint_rotation_channels.append((channels[0], channels[1], channels[2]))
        else:
            print('cannot find joint channels')
        self.line_number += 1

        while self.lines[self.line_number].startswith('JOINT') or \
                self.lines[self.line_number].startswith('End'):
            if self.lines[self.line_number].startswith('JOINT'):
                self.parse_joint(index)
            elif self.lines[self.line_number].startswith('End'):
                self.parse_end(index)
        self.line_number += 1

    def parse_end(self, parent):
        self.joint_parents.append(parent)

        self.joint_names.append(self.joint_names[parent] + '_end')
        self.line_number += 2

        if self.lines[self.line_number].startswith('OFFSET'):
            self.joint_offsets.append(self.parse_offset(self.lines[self.line_number]))
        else:
            print('cannot find joint offset')
        self.line_number += 2

    def analyze(self):
        if not self.lines[self.line_number].startswith('HIERARCHY'):
            print('cannot find hierarchy')
        self.line_number += 1

        if self.lines[self.line_number].startswith('ROOT'):
            self.parse_root()
    
        return self.joint_names, self.joint_parents, self.joint_offsets


def forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    m = len(joint_name)
    joint_positions = np.zeros((m, 3), dtype=np.float64)
    joint_orientations = np.zeros((m, 4), dtype=np.float64)
    channels = motion_data[frame_id]
    rotations = np.zeros((m, 3), dtype=np.float64)
    cnt = 1
    for i in range(m):
        if '_end' not in joint_name[i]:
            for j in range(3):
                rotations[i][j] = channels[cnt * 3 + j]
            cnt += 1
    for i in range(m):
        parent = joint_parent[i]
        if parent == -1:
            for j in range(3):
                joint_positions[0][j] = channels[j]
            joint_orientations[0] = R.from_euler('XYZ', [rotations[0][0], rotations[0][1], rotations[0][2]], degrees=True).as_quat()
        else:
            if '_end' in joint_name[i]:
                joint_orientations[i] = np.array([0, 0, 0, 1])
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).as_matrix() @ joint_offset[i]
            else:
                rotation = R.from_euler('XYZ', [rotations[i][0], rotations[i][1], rotations[i][2]], degrees=True)
                joint_orientations[i] = (R.from_quat(joint_orientations[parent]) * rotation).as_quat()
                joint_positions[i] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).as_matrix() @ joint_offset[i]

    return joint_positions, joint_orientations


def load_motion_data(bvh_file_path):
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def animation(viewer, joint_names, joint_parents, joint_offsets, motion_data):
    frame_num = motion_data.shape[0]
    
    class UpdateHandle:
        def __init__(self):
            self.current_frame = 0
        def update_func(self, viewer_):
            joint_positions, joint_orientations = forward_kinematics(joint_names, \
                joint_parents, joint_offsets, motion_data, self.current_frame)
            viewer.show_pose(joint_names, joint_positions, joint_orientations)
            self.current_frame = (self.current_frame + 1) % frame_num

    handle = UpdateHandle()
    viewer.update_func = handle.update_func
    viewer.run()


def main():
    bvh_file_path = 'walk60.bvh'
    viewer = SimpleViewer()
    parser = HierarchyParser(bvh_file_path)
    joint_names, joint_parents, joint_offsets = parser.analyze()
    motion_data = load_motion_data(bvh_file_path)
    animation(viewer, joint_names, joint_parents, joint_offsets, motion_data)


if __name__ == "__main__":
    main()
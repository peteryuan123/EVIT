import rosbag
import sys

def main():
    bag_name = sys.argv[1]
    file_name = sys.argv[2]

    bag = rosbag.Bag(bag_name, "r")
    gt_data = bag.read_messages("/gt/pose")

    with open(file_name, "w") as f:
        for topic, msg, t in gt_data:
            str_t = str(t)
            print(str_t[:10] + "." + str_t[10:15], end=" ", file=f)
            tx, ty, tz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
            qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
            print(str(tx) + " " + str(ty) + " " + str(tz), end=" ", file=f)
            print(str(qx) + " " + str(qy) + " " + str(qz) + " " + str(qw), end="\n", file=f)



if __name__ == '__main__':
    main()
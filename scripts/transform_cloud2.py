import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as sciR
import cv2

# cloud_path = "/home/cuili/zzz/Canny_EVIT-main/SemiDense1.pcd"
cloud_path = "/home/mpl/Downloads/pc.txt"

def invT(T):
    inv = np.eye(4)
    inv[:3, :3] = T[:3,:3].T
    inv[:3, 3] = (-T[:3,:3].T @ T[:3, 3].reshape(3, 1)).flatten()
    return inv


def vector2Transformation(vector):
    T = np.eye(4)
    T[:3, 3] = np.array(vector[:3])
    T[:3, :3] = sciR.from_quat([vector[3:]]).as_matrix()
    return T

cloud = np.loadtxt(cloud_path)
scale = 0.9420262101845873
cloud = cloud[:, :6]
cloud = np.multiply(cloud, scale)

T_cam_body = np.array([-0.857137023976571  ,  0.03276713258773897, -0.5140451703406658 ,  0.09127742788053987,
                        0.01322063096422759, -0.9962462506036175 , -0.08554895133864114, -0.02255409664008403,
                        -0.5149187674240416 , -0.08012317505073682,  0.853486344222504  , -0.02986309837992267,
                        0                  ,  0                  ,  0                  ,  1                  ]).reshape(4,4)
T_cam_imu = np.array([[0.017248643674008135, -0.9998037138739959, 0.009747718459772736, 0.07733078169916466],
                        [0.012834636469124028, -0.009526963092989282, -0.999872246379971, -0.016637889364465353],
                        [0.9997688514842376, 0.017371548520172697, 0.01266779001636642, -0.14481844113148515],
                        [0.0, 0.0, 0.0, 1.0]])

T_cam_event = np.array([0.9999407352369797  ,  0.009183655542749752, 0.005846920950435052 ,  0.0005085820608404798,
                        -0.009131364645448854, 0.9999186289230431 , -0.008908070070089353, -0.04081979450823404,
                        -0.005928253827254812 , 0.008854151768176144,  0.9999432282899994  , -0.0140781304960408,
                        0                  ,  0                  ,  0                  ,  1                  ]).reshape(4,4)


T_ENU_imuAt41_63 = vector2Transformation(np.array([ 0.06955,0.05161,0.15389,0.98909,-0.01123,0.13796,-0.05037]))
T_orb_camAt38_93 = vector2Transformation(np.array([  0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000]))
T_mocap_bodyAt41_63 = vector2Transformation(np.array([ 0.042883, 1.533522, 1.275031, -0.085174, -0.307555, -0.016232, -0.947572]))
T_mocap_bodyAt38_93 = vector2Transformation(np.array([ -0.013187, 1.364460, 1.205243, -0.112441, -0.366111, -0.008659, -0.923712]))


T_enu_orb = T_ENU_imuAt41_63 @ invT(T_cam_imu) @ T_cam_body @ invT(T_mocap_bodyAt41_63) @ T_mocap_bodyAt38_93 @ invT(T_cam_body)
cloud[:, :3] = (T_enu_orb[:3, :3] @ cloud[:, :3].transpose() + T_enu_orb[:3, 3].reshape(3, 1)).transpose()
cloud[:, 3:6] = (T_enu_orb[:3, :3] @ cloud[:, 3:6].transpose() + T_enu_orb[:3, 3].reshape(3, 1)).transpose()

print(T_enu_orb @ T_cam_imu)
print(invT(T_cam_imu) @ T_cam_event)
np.savetxt("/home/mpl/data/EVIT/cloud/Robot.txt", cloud)

#img = cv2.imread("/home/cuili/zzz/scripts/result/vector/robot_n/rgb/1642661138.936530.png")
img = cv2.imread("/home/mpl/data/EVIT/robot/robot_normal_result/robot_normal_1642661138.936530.jpg")


distortion_coeffs = np.array(
    [
        -0.315760, 0.104955, 0.000320, -0.000156, 0.000000
    ]
)


matrix_camera = np.array(
    [[886.19107, 0, 610.57891], [0, 886.59163, 514.59271], [0, 0, 1]],
    dtype="double",
)


#[ 327.32749,    0.     ,  304.97749,
 #           0.     ,  327.46184,  235.37621,
  #          0.     ,    0.     ,    1.     ]event K

#T_enu_event = T_enu_orb @ T_cam_event
T_enu_event = (T_enu_orb @ T_cam_imu) @ (invT(T_cam_imu) @ T_cam_event)
print("Twb:")
print((T_enu_orb @ T_cam_imu))

print("Tbc:")
print(invT(T_cam_imu) @ T_cam_event)

print("Twc:")
print(T_enu_event)
T_event_enu = invT(T_enu_event)

cloud = np.loadtxt("/home/mpl/data/EVIT/cloud/Robot.txt")

print("------")
# print(T_enu_orb @ T_cam_imu)
# print(invT(T_cam_imu) @ T_cam_event)
print(T_event_enu)
pc = (T_event_enu[:3, :3] @ cloud[:, :3].transpose() + T_event_enu[:3, 3].reshape(3, 1)).transpose()

for p in pc:
    #x = (p[0] * 886.19107 + p[2] * 610.57891) / p[2]
    #y = (p[1] * 886.59163 + p[2] * 514.59271) / p[2]
    x = (p[0] * 327.32749 + p[2] * 304.97749) / p[2]
    y = (p[1] * 327.46184 + p[2] * 235.37621) / p[2]
    cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), -1)

cv2.imshow("Final", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

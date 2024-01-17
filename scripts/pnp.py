import cv2
import numpy as np

image_points_2D = np.array(
    [
        (864, 486),
        (809, 762),
        (615, 755),
        (1079, 890),
        (684, 378),
        (391, 481),
    ],
    dtype="double",
)

figure_points_3D = np.array(
    [
        (1.214223, -0.221452, -0.599522),
        (0.762521, 0.123064, -0.574197),
        (0.810329, 0.292683, -0.565690),
        (0.575259, -0.090543,-0.605335),
        (0.993210, 0.196819, -0.319163),
        (1.331182, 0.404702, -0.603304),
    ]
)

distortion_coeffs = np.array(
    [
        -0.315760, 0.104955, 0.000320, -0.000156, 0.000000
    ]
)

matrix_camera = np.array(
    [[886.19107, 0, 610.57891], [0, 886.59163, 514.59271], [0, 0, 1]],
    dtype="double",
)
success, vector_rotation, vector_translation = cv2.solvePnP(
    figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=0
)

R_cam_w,_ = cv2.Rodrigues(vector_rotation)
T_w_cam = np.eye(4)
T_w_cam[:3, :3] = R_cam_w.T
T_w_cam[:3, 3] = (-R_cam_w.T @ vector_translation.reshape(3, 1)).flatten()

# T_w_cam[:3, :3] = R_cam_w
# T_w_cam[:3, 3] = vector_translation.flatten()

T_cam_event = np.array([ 0.9999407352369797  , 0.009183655542749752,  0.005846920950435052,  0.0005085820608404798,
                         -0.009131364645448854, 0.9999186289230431  , -0.008908070070089353, -0.04081979450823404  ,
                         -0.005928253827254812, 0.008854151768176144,  0.9999432282899994  , -0.0140781304960408   ,
                         0                   , 0                   ,  0                   ,  1                    ]).reshape(4,4)

T_cam_imu = np.array([0.017248643674008135, -0.9998037138739959, 0.009747718459772736, 0.07733078169916466,
                      0.012834636469124028, -0.009526963092989282, -0.999872246379971, -0.016637889364465353,
                      0.9997688514842376, 0.017371548520172697, 0.01266779001636642, -0.14481844113148515,
                      0.0, 0.0, 0.0, 1.0]).reshape(4,4)


T_w_imu = T_w_cam @ T_cam_imu


for i in range(3):
    for j in range(3):
        print(T_w_imu[i, j], end=", ")
    print("")
for i in range(3):
    print(T_w_imu[i, 3], end=", ")
print("")

print("R0:")
print(T_w_imu[:3, :3])
print("t0:")
print(T_w_imu[:3, 3])

img = cv2.imread("/home/mpl/data/EVIT/PnP/robot_fast/1642661813.131114721.jpg") #img

pc = np.loadtxt('/home/mpl/data/EVIT/cloud/Robot.txt') #point cloud
pc = pc[:, :3]
for p in pc:
    nose_end_point2D, jacobian = cv2.projectPoints(
    p,
    vector_rotation,
    vector_translation,
    matrix_camera,
    distortion_coeffs,
    )
    cv2.circle(img, (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])), 3, (0, 255, 255), -1)

for p in image_points_2D:
    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

print(vector_rotation)
print(vector_translation)




cv2.imshow("Final", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

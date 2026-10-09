# Feature index table (238 features)

Source: `utili_Getpose_v4.py` → `GetAllFeatures()` concatenation order:
`embed_coors` (34) + `dis_embedders` (106) + `ang_embedders` (32) + `gait_phases` (66).

| Index | Block | Code | Meaning |
|------:|-------|------|---------|
| 0 | embed_coors | `norm_coord_nose_x` | Normalized X of landmark 'nose' (YOLOv8 index 0); hip-centered, scale-normalized, ×1e5 |
| 1 | embed_coors | `norm_coord_nose_y` | Normalized Y of landmark 'nose' (YOLOv8 index 0); hip-centered, scale-normalized, ×1e5 |
| 2 | embed_coors | `norm_coord_left_eye_x` | Normalized X of landmark 'left_eye' (YOLOv8 index 1); hip-centered, scale-normalized, ×1e5 |
| 3 | embed_coors | `norm_coord_left_eye_y` | Normalized Y of landmark 'left_eye' (YOLOv8 index 1); hip-centered, scale-normalized, ×1e5 |
| 4 | embed_coors | `norm_coord_right_eye_x` | Normalized X of landmark 'right_eye' (YOLOv8 index 2); hip-centered, scale-normalized, ×1e5 |
| 5 | embed_coors | `norm_coord_right_eye_y` | Normalized Y of landmark 'right_eye' (YOLOv8 index 2); hip-centered, scale-normalized, ×1e5 |
| 6 | embed_coors | `norm_coord_left_ear_x` | Normalized X of landmark 'left_ear' (YOLOv8 index 3); hip-centered, scale-normalized, ×1e5 |
| 7 | embed_coors | `norm_coord_left_ear_y` | Normalized Y of landmark 'left_ear' (YOLOv8 index 3); hip-centered, scale-normalized, ×1e5 |
| 8 | embed_coors | `norm_coord_right_ear_x` | Normalized X of landmark 'right_ear' (YOLOv8 index 4); hip-centered, scale-normalized, ×1e5 |
| 9 | embed_coors | `norm_coord_right_ear_y` | Normalized Y of landmark 'right_ear' (YOLOv8 index 4); hip-centered, scale-normalized, ×1e5 |
| 10 | embed_coors | `norm_coord_left_shoulder_x` | Normalized X of landmark 'left_shoulder' (YOLOv8 index 5); hip-centered, scale-normalized, ×1e5 |
| 11 | embed_coors | `norm_coord_left_shoulder_y` | Normalized Y of landmark 'left_shoulder' (YOLOv8 index 5); hip-centered, scale-normalized, ×1e5 |
| 12 | embed_coors | `norm_coord_right_shoulder_x` | Normalized X of landmark 'right_shoulder' (YOLOv8 index 6); hip-centered, scale-normalized, ×1e5 |
| 13 | embed_coors | `norm_coord_right_shoulder_y` | Normalized Y of landmark 'right_shoulder' (YOLOv8 index 6); hip-centered, scale-normalized, ×1e5 |
| 14 | embed_coors | `norm_coord_left_elbow_x` | Normalized X of landmark 'left_elbow' (YOLOv8 index 7); hip-centered, scale-normalized, ×1e5 |
| 15 | embed_coors | `norm_coord_left_elbow_y` | Normalized Y of landmark 'left_elbow' (YOLOv8 index 7); hip-centered, scale-normalized, ×1e5 |
| 16 | embed_coors | `norm_coord_right_elbow_x` | Normalized X of landmark 'right_elbow' (YOLOv8 index 8); hip-centered, scale-normalized, ×1e5 |
| 17 | embed_coors | `norm_coord_right_elbow_y` | Normalized Y of landmark 'right_elbow' (YOLOv8 index 8); hip-centered, scale-normalized, ×1e5 |
| 18 | embed_coors | `norm_coord_left_wrist_x` | Normalized X of landmark 'left_wrist' (YOLOv8 index 9); hip-centered, scale-normalized, ×1e5 |
| 19 | embed_coors | `norm_coord_left_wrist_y` | Normalized Y of landmark 'left_wrist' (YOLOv8 index 9); hip-centered, scale-normalized, ×1e5 |
| 20 | embed_coors | `norm_coord_right_wrist_x` | Normalized X of landmark 'right_wrist' (YOLOv8 index 10); hip-centered, scale-normalized, ×1e5 |
| 21 | embed_coors | `norm_coord_right_wrist_y` | Normalized Y of landmark 'right_wrist' (YOLOv8 index 10); hip-centered, scale-normalized, ×1e5 |
| 22 | embed_coors | `norm_coord_left_hip_x` | Normalized X of landmark 'left_hip' (YOLOv8 index 11); hip-centered, scale-normalized, ×1e5 |
| 23 | embed_coors | `norm_coord_left_hip_y` | Normalized Y of landmark 'left_hip' (YOLOv8 index 11); hip-centered, scale-normalized, ×1e5 |
| 24 | embed_coors | `norm_coord_right_hip_x` | Normalized X of landmark 'right_hip' (YOLOv8 index 12); hip-centered, scale-normalized, ×1e5 |
| 25 | embed_coors | `norm_coord_right_hip_y` | Normalized Y of landmark 'right_hip' (YOLOv8 index 12); hip-centered, scale-normalized, ×1e5 |
| 26 | embed_coors | `norm_coord_left_knee_x` | Normalized X of landmark 'left_knee' (YOLOv8 index 13); hip-centered, scale-normalized, ×1e5 |
| 27 | embed_coors | `norm_coord_left_knee_y` | Normalized Y of landmark 'left_knee' (YOLOv8 index 13); hip-centered, scale-normalized, ×1e5 |
| 28 | embed_coors | `norm_coord_right_knee_x` | Normalized X of landmark 'right_knee' (YOLOv8 index 14); hip-centered, scale-normalized, ×1e5 |
| 29 | embed_coors | `norm_coord_right_knee_y` | Normalized Y of landmark 'right_knee' (YOLOv8 index 14); hip-centered, scale-normalized, ×1e5 |
| 30 | embed_coors | `norm_coord_left_ankle_x` | Normalized X of landmark 'left_ankle' (YOLOv8 index 15); hip-centered, scale-normalized, ×1e5 |
| 31 | embed_coors | `norm_coord_left_ankle_y` | Normalized Y of landmark 'left_ankle' (YOLOv8 index 15); hip-centered, scale-normalized, ×1e5 |
| 32 | embed_coors | `norm_coord_right_ankle_x` | Normalized X of landmark 'right_ankle' (YOLOv8 index 16); hip-centered, scale-normalized, ×1e5 |
| 33 | embed_coors | `norm_coord_right_ankle_y` | Normalized Y of landmark 'right_ankle' (YOLOv8 index 16); hip-centered, scale-normalized, ×1e5 |
| 34 | dis_embedders | `dis_00_x` | Signed DX of vector from 'avg(left_hip,right_hip)' to 'avg(left_shoulder,right_shoulder)' (pair 0 in distance embedding) |
| 35 | dis_embedders | `dis_00_y` | Signed DY of vector from 'avg(left_hip,right_hip)' to 'avg(left_shoulder,right_shoulder)' (pair 0 in distance embedding) |
| 36 | dis_embedders | `dis_01_x` | Signed DX of vector from 'nose' to 'left_shoulder' (pair 1 in distance embedding) |
| 37 | dis_embedders | `dis_01_y` | Signed DY of vector from 'nose' to 'left_shoulder' (pair 1 in distance embedding) |
| 38 | dis_embedders | `dis_02_x` | Signed DX of vector from 'nose' to 'right_shoulder' (pair 2 in distance embedding) |
| 39 | dis_embedders | `dis_02_y` | Signed DY of vector from 'nose' to 'right_shoulder' (pair 2 in distance embedding) |
| 40 | dis_embedders | `dis_03_x` | Signed DX of vector from 'nose' to 'left_hip' (pair 3 in distance embedding) |
| 41 | dis_embedders | `dis_03_y` | Signed DY of vector from 'nose' to 'left_hip' (pair 3 in distance embedding) |
| 42 | dis_embedders | `dis_04_x` | Signed DX of vector from 'nose' to 'right_hip' (pair 4 in distance embedding) |
| 43 | dis_embedders | `dis_04_y` | Signed DY of vector from 'nose' to 'right_hip' (pair 4 in distance embedding) |
| 44 | dis_embedders | `dis_05_x` | Signed DX of vector from 'left_elbow' to 'left_shoulder' (pair 5 in distance embedding) |
| 45 | dis_embedders | `dis_05_y` | Signed DY of vector from 'left_elbow' to 'left_shoulder' (pair 5 in distance embedding) |
| 46 | dis_embedders | `dis_06_x` | Signed DX of vector from 'left_elbow' to 'left_hip' (pair 6 in distance embedding) |
| 47 | dis_embedders | `dis_06_y` | Signed DY of vector from 'left_elbow' to 'left_hip' (pair 6 in distance embedding) |
| 48 | dis_embedders | `dis_07_x` | Signed DX of vector from 'left_elbow' to 'left_knee' (pair 7 in distance embedding) |
| 49 | dis_embedders | `dis_07_y` | Signed DY of vector from 'left_elbow' to 'left_knee' (pair 7 in distance embedding) |
| 50 | dis_embedders | `dis_08_x` | Signed DX of vector from 'left_elbow' to 'left_ankle' (pair 8 in distance embedding) |
| 51 | dis_embedders | `dis_08_y` | Signed DY of vector from 'left_elbow' to 'left_ankle' (pair 8 in distance embedding) |
| 52 | dis_embedders | `dis_09_x` | Signed DX of vector from 'left_elbow' to 'right_shoulder' (pair 9 in distance embedding) |
| 53 | dis_embedders | `dis_09_y` | Signed DY of vector from 'left_elbow' to 'right_shoulder' (pair 9 in distance embedding) |
| 54 | dis_embedders | `dis_10_x` | Signed DX of vector from 'left_elbow' to 'right_hip' (pair 10 in distance embedding) |
| 55 | dis_embedders | `dis_10_y` | Signed DY of vector from 'left_elbow' to 'right_hip' (pair 10 in distance embedding) |
| 56 | dis_embedders | `dis_11_x` | Signed DX of vector from 'left_elbow' to 'right_knee' (pair 11 in distance embedding) |
| 57 | dis_embedders | `dis_11_y` | Signed DY of vector from 'left_elbow' to 'right_knee' (pair 11 in distance embedding) |
| 58 | dis_embedders | `dis_12_x` | Signed DX of vector from 'left_elbow' to 'right_ankle' (pair 12 in distance embedding) |
| 59 | dis_embedders | `dis_12_y` | Signed DY of vector from 'left_elbow' to 'right_ankle' (pair 12 in distance embedding) |
| 60 | dis_embedders | `dis_13_x` | Signed DX of vector from 'right_elbow' to 'left_shoulder' (pair 13 in distance embedding) |
| 61 | dis_embedders | `dis_13_y` | Signed DY of vector from 'right_elbow' to 'left_shoulder' (pair 13 in distance embedding) |
| 62 | dis_embedders | `dis_14_x` | Signed DX of vector from 'right_elbow' to 'left_hip' (pair 14 in distance embedding) |
| 63 | dis_embedders | `dis_14_y` | Signed DY of vector from 'right_elbow' to 'left_hip' (pair 14 in distance embedding) |
| 64 | dis_embedders | `dis_15_x` | Signed DX of vector from 'right_elbow' to 'left_knee' (pair 15 in distance embedding) |
| 65 | dis_embedders | `dis_15_y` | Signed DY of vector from 'right_elbow' to 'left_knee' (pair 15 in distance embedding) |
| 66 | dis_embedders | `dis_16_x` | Signed DX of vector from 'right_elbow' to 'left_ankle' (pair 16 in distance embedding) |
| 67 | dis_embedders | `dis_16_y` | Signed DY of vector from 'right_elbow' to 'left_ankle' (pair 16 in distance embedding) |
| 68 | dis_embedders | `dis_17_x` | Signed DX of vector from 'right_elbow' to 'right_shoulder' (pair 17 in distance embedding) |
| 69 | dis_embedders | `dis_17_y` | Signed DY of vector from 'right_elbow' to 'right_shoulder' (pair 17 in distance embedding) |
| 70 | dis_embedders | `dis_18_x` | Signed DX of vector from 'right_elbow' to 'right_hip' (pair 18 in distance embedding) |
| 71 | dis_embedders | `dis_18_y` | Signed DY of vector from 'right_elbow' to 'right_hip' (pair 18 in distance embedding) |
| 72 | dis_embedders | `dis_19_x` | Signed DX of vector from 'right_elbow' to 'right_knee' (pair 19 in distance embedding) |
| 73 | dis_embedders | `dis_19_y` | Signed DY of vector from 'right_elbow' to 'right_knee' (pair 19 in distance embedding) |
| 74 | dis_embedders | `dis_20_x` | Signed DX of vector from 'right_elbow' to 'right_ankle' (pair 20 in distance embedding) |
| 75 | dis_embedders | `dis_20_y` | Signed DY of vector from 'right_elbow' to 'right_ankle' (pair 20 in distance embedding) |
| 76 | dis_embedders | `dis_21_x` | Signed DX of vector from 'left_shoulder' to 'left_hip' (pair 21 in distance embedding) |
| 77 | dis_embedders | `dis_21_y` | Signed DY of vector from 'left_shoulder' to 'left_hip' (pair 21 in distance embedding) |
| 78 | dis_embedders | `dis_22_x` | Signed DX of vector from 'left_shoulder' to 'left_knee' (pair 22 in distance embedding) |
| 79 | dis_embedders | `dis_22_y` | Signed DY of vector from 'left_shoulder' to 'left_knee' (pair 22 in distance embedding) |
| 80 | dis_embedders | `dis_23_x` | Signed DX of vector from 'left_shoulder' to 'left_ankle' (pair 23 in distance embedding) |
| 81 | dis_embedders | `dis_23_y` | Signed DY of vector from 'left_shoulder' to 'left_ankle' (pair 23 in distance embedding) |
| 82 | dis_embedders | `dis_24_x` | Signed DX of vector from 'left_shoulder' to 'right_hip' (pair 24 in distance embedding) |
| 83 | dis_embedders | `dis_24_y` | Signed DY of vector from 'left_shoulder' to 'right_hip' (pair 24 in distance embedding) |
| 84 | dis_embedders | `dis_25_x` | Signed DX of vector from 'left_shoulder' to 'right_knee' (pair 25 in distance embedding) |
| 85 | dis_embedders | `dis_25_y` | Signed DY of vector from 'left_shoulder' to 'right_knee' (pair 25 in distance embedding) |
| 86 | dis_embedders | `dis_26_x` | Signed DX of vector from 'left_shoulder' to 'right_ankle' (pair 26 in distance embedding) |
| 87 | dis_embedders | `dis_26_y` | Signed DY of vector from 'left_shoulder' to 'right_ankle' (pair 26 in distance embedding) |
| 88 | dis_embedders | `dis_27_x` | Signed DX of vector from 'right_shoulder' to 'left_hip' (pair 27 in distance embedding) |
| 89 | dis_embedders | `dis_27_y` | Signed DY of vector from 'right_shoulder' to 'left_hip' (pair 27 in distance embedding) |
| 90 | dis_embedders | `dis_28_x` | Signed DX of vector from 'right_shoulder' to 'left_knee' (pair 28 in distance embedding) |
| 91 | dis_embedders | `dis_28_y` | Signed DY of vector from 'right_shoulder' to 'left_knee' (pair 28 in distance embedding) |
| 92 | dis_embedders | `dis_29_x` | Signed DX of vector from 'right_shoulder' to 'left_ankle' (pair 29 in distance embedding) |
| 93 | dis_embedders | `dis_29_y` | Signed DY of vector from 'right_shoulder' to 'left_ankle' (pair 29 in distance embedding) |
| 94 | dis_embedders | `dis_30_x` | Signed DX of vector from 'right_shoulder' to 'right_hip' (pair 30 in distance embedding) |
| 95 | dis_embedders | `dis_30_y` | Signed DY of vector from 'right_shoulder' to 'right_hip' (pair 30 in distance embedding) |
| 96 | dis_embedders | `dis_31_x` | Signed DX of vector from 'right_shoulder' to 'right_knee' (pair 31 in distance embedding) |
| 97 | dis_embedders | `dis_31_y` | Signed DY of vector from 'right_shoulder' to 'right_knee' (pair 31 in distance embedding) |
| 98 | dis_embedders | `dis_32_x` | Signed DX of vector from 'right_shoulder' to 'right_ankle' (pair 32 in distance embedding) |
| 99 | dis_embedders | `dis_32_y` | Signed DY of vector from 'right_shoulder' to 'right_ankle' (pair 32 in distance embedding) |
| 100 | dis_embedders | `dis_33_x` | Signed DX of vector from 'left_hip' to 'left_knee' (pair 33 in distance embedding) |
| 101 | dis_embedders | `dis_33_y` | Signed DY of vector from 'left_hip' to 'left_knee' (pair 33 in distance embedding) |
| 102 | dis_embedders | `dis_34_x` | Signed DX of vector from 'left_hip' to 'left_ankle' (pair 34 in distance embedding) |
| 103 | dis_embedders | `dis_34_y` | Signed DY of vector from 'left_hip' to 'left_ankle' (pair 34 in distance embedding) |
| 104 | dis_embedders | `dis_35_x` | Signed DX of vector from 'left_hip' to 'right_knee' (pair 35 in distance embedding) |
| 105 | dis_embedders | `dis_35_y` | Signed DY of vector from 'left_hip' to 'right_knee' (pair 35 in distance embedding) |
| 106 | dis_embedders | `dis_36_x` | Signed DX of vector from 'left_hip' to 'right_ankle' (pair 36 in distance embedding) |
| 107 | dis_embedders | `dis_36_y` | Signed DY of vector from 'left_hip' to 'right_ankle' (pair 36 in distance embedding) |
| 108 | dis_embedders | `dis_37_x` | Signed DX of vector from 'right_hip' to 'left_knee' (pair 37 in distance embedding) |
| 109 | dis_embedders | `dis_37_y` | Signed DY of vector from 'right_hip' to 'left_knee' (pair 37 in distance embedding) |
| 110 | dis_embedders | `dis_38_x` | Signed DX of vector from 'right_hip' to 'left_ankle' (pair 38 in distance embedding) |
| 111 | dis_embedders | `dis_38_y` | Signed DY of vector from 'right_hip' to 'left_ankle' (pair 38 in distance embedding) |
| 112 | dis_embedders | `dis_39_x` | Signed DX of vector from 'right_hip' to 'right_knee' (pair 39 in distance embedding) |
| 113 | dis_embedders | `dis_39_y` | Signed DY of vector from 'right_hip' to 'right_knee' (pair 39 in distance embedding) |
| 114 | dis_embedders | `dis_40_x` | Signed DX of vector from 'right_hip' to 'right_ankle' (pair 40 in distance embedding) |
| 115 | dis_embedders | `dis_40_y` | Signed DY of vector from 'right_hip' to 'right_ankle' (pair 40 in distance embedding) |
| 116 | dis_embedders | `dis_41_x` | Signed DX of vector from 'left_wrist' to 'right_wrist' (pair 41 in distance embedding) |
| 117 | dis_embedders | `dis_41_y` | Signed DY of vector from 'left_wrist' to 'right_wrist' (pair 41 in distance embedding) |
| 118 | dis_embedders | `dis_42_x` | Signed DX of vector from 'left_elbow' to 'right_elbow' (pair 42 in distance embedding) |
| 119 | dis_embedders | `dis_42_y` | Signed DY of vector from 'left_elbow' to 'right_elbow' (pair 42 in distance embedding) |
| 120 | dis_embedders | `dis_43_x` | Signed DX of vector from 'left_shoulder' to 'right_shoulder' (pair 43 in distance embedding) |
| 121 | dis_embedders | `dis_43_y` | Signed DY of vector from 'left_shoulder' to 'right_shoulder' (pair 43 in distance embedding) |
| 122 | dis_embedders | `dis_44_x` | Signed DX of vector from 'left_hip' to 'right_hip' (pair 44 in distance embedding) |
| 123 | dis_embedders | `dis_44_y` | Signed DY of vector from 'left_hip' to 'right_hip' (pair 44 in distance embedding) |
| 124 | dis_embedders | `dis_45_x` | Signed DX of vector from 'left_knee' to 'right_knee' (pair 45 in distance embedding) |
| 125 | dis_embedders | `dis_45_y` | Signed DY of vector from 'left_knee' to 'right_knee' (pair 45 in distance embedding) |
| 126 | dis_embedders | `dis_46_x` | Signed DX of vector from 'left_ankle' to 'right_ankle' (pair 46 in distance embedding) |
| 127 | dis_embedders | `dis_46_y` | Signed DY of vector from 'left_ankle' to 'right_ankle' (pair 46 in distance embedding) |
| 128 | dis_embedders | `dis_47_x` | Signed DX of vector from 'avg(left_hip,right_hip)' to 'avg(right_ankle,right_ankle)' (pair 47 in distance embedding) |
| 129 | dis_embedders | `dis_47_y` | Signed DY of vector from 'avg(left_hip,right_hip)' to 'avg(right_ankle,right_ankle)' (pair 47 in distance embedding) |
| 130 | dis_embedders | `dis_48_x` | Signed DX of vector from 'avg(left_hip,right_hip)' to 'avg(left_ankle,left_ankle)' (pair 48 in distance embedding) |
| 131 | dis_embedders | `dis_48_y` | Signed DY of vector from 'avg(left_hip,right_hip)' to 'avg(left_ankle,left_ankle)' (pair 48 in distance embedding) |
| 132 | dis_embedders | `dis_49_x` | Signed DX of vector from 'avg(left_hip,right_hip)' to 'avg(right_shoulder,right_shoulder)' (pair 49 in distance embedding) |
| 133 | dis_embedders | `dis_49_y` | Signed DY of vector from 'avg(left_hip,right_hip)' to 'avg(right_shoulder,right_shoulder)' (pair 49 in distance embedding) |
| 134 | dis_embedders | `dis_50_x` | Signed DX of vector from 'avg(left_hip,right_hip)' to 'avg(left_shoulder,left_shoulder)' (pair 50 in distance embedding) |
| 135 | dis_embedders | `dis_50_y` | Signed DY of vector from 'avg(left_hip,right_hip)' to 'avg(left_shoulder,left_shoulder)' (pair 50 in distance embedding) |
| 136 | dis_embedders | `dis_51_x` | Signed DX of vector from 'avg(left_eye,right_eye)' to 'avg(right_shoulder,right_shoulder)' (pair 51 in distance embedding) |
| 137 | dis_embedders | `dis_51_y` | Signed DY of vector from 'avg(left_eye,right_eye)' to 'avg(right_shoulder,right_shoulder)' (pair 51 in distance embedding) |
| 138 | dis_embedders | `dis_52_x` | Signed DX of vector from 'avg(left_eye,right_eye)' to 'avg(left_shoulder,left_shoulder)' (pair 52 in distance embedding) |
| 139 | dis_embedders | `dis_52_y` | Signed DY of vector from 'avg(left_eye,right_eye)' to 'avg(left_shoulder,left_shoulder)' (pair 52 in distance embedding) |
| 140 | ang_embedders | `ang_00` | Signed angle (degrees) between vector left_shoulder→right_shoulder and left_hip→right_hip |
| 141 | ang_embedders | `ang_01` | Signed angle (degrees) between vector left_elbow→left_shoulder and right_elbow→right_shoulder |
| 142 | ang_embedders | `ang_02` | Signed angle (degrees) between vector left_elbow→left_shoulder and right_hip→right_shoulder |
| 143 | ang_embedders | `ang_03` | Signed angle (degrees) between vector left_elbow→left_shoulder and right_knee→right_hip |
| 144 | ang_embedders | `ang_04` | Signed angle (degrees) between vector left_elbow→left_shoulder and right_ankle→right_knee |
| 145 | ang_embedders | `ang_05` | Signed angle (degrees) between vector left_elbow→left_shoulder and left_hip→left_shoulder |
| 146 | ang_embedders | `ang_06` | Signed angle (degrees) between vector left_elbow→left_shoulder and left_knee→left_hip |
| 147 | ang_embedders | `ang_07` | Signed angle (degrees) between vector left_elbow→left_shoulder and left_ankle→left_knee |
| 148 | ang_embedders | `ang_08` | Signed angle (degrees) between vector right_elbow→right_shoulder and right_hip→right_shoulder |
| 149 | ang_embedders | `ang_09` | Signed angle (degrees) between vector right_elbow→right_shoulder and right_knee→right_hip |
| 150 | ang_embedders | `ang_10` | Signed angle (degrees) between vector right_elbow→right_shoulder and right_ankle→right_knee |
| 151 | ang_embedders | `ang_11` | Signed angle (degrees) between vector right_elbow→right_shoulder and left_hip→left_shoulder |
| 152 | ang_embedders | `ang_12` | Signed angle (degrees) between vector right_elbow→right_shoulder and left_knee→left_hip |
| 153 | ang_embedders | `ang_13` | Signed angle (degrees) between vector right_elbow→right_shoulder and left_ankle→left_knee |
| 154 | ang_embedders | `ang_14` | Signed angle (degrees) between vector left_shoulder→left_hip and right_shoulder→right_hip |
| 155 | ang_embedders | `ang_15` | Signed angle (degrees) between vector left_shoulder→left_hip and right_knee→right_hip |
| 156 | ang_embedders | `ang_16` | Signed angle (degrees) between vector left_shoulder→left_hip and right_ankle→right_knee |
| 157 | ang_embedders | `ang_17` | Signed angle (degrees) between vector left_shoulder→left_hip and left_knee→left_hip |
| 158 | ang_embedders | `ang_18` | Signed angle (degrees) between vector left_shoulder→left_hip and left_ankle→left_knee |
| 159 | ang_embedders | `ang_19` | Signed angle (degrees) between vector right_shoulder→right_hip and right_knee→right_hip |
| 160 | ang_embedders | `ang_20` | Signed angle (degrees) between vector right_shoulder→right_hip and right_ankle→right_knee |
| 161 | ang_embedders | `ang_21` | Signed angle (degrees) between vector right_shoulder→right_hip and left_knee→left_hip |
| 162 | ang_embedders | `ang_22` | Signed angle (degrees) between vector right_shoulder→right_hip and left_ankle→left_knee |
| 163 | ang_embedders | `ang_23` | Signed angle (degrees) between vector left_hip→left_knee and right_hip→right_knee |
| 164 | ang_embedders | `ang_24` | Signed angle (degrees) between vector left_hip→left_knee and right_ankle→right_knee |
| 165 | ang_embedders | `ang_25` | Signed angle (degrees) between vector left_hip→left_knee and left_ankle→left_knee |
| 166 | ang_embedders | `ang_26` | Signed angle (degrees) between vector right_hip→right_knee and right_ankle→right_knee |
| 167 | ang_embedders | `ang_27` | Signed angle (degrees) between vector right_hip→right_knee and left_ankle→left_knee |
| 168 | ang_embedders | `ang_28` | Signed angle (degrees) between vector left_hip→right_elbow and left_knee→right_elbow |
| 169 | ang_embedders | `ang_29` | Signed angle (degrees) between vector right_hip→right_elbow and right_knee→right_elbow |
| 170 | ang_embedders | `ang_30` | Signed angle (degrees) between vector right_hip→left_elbow and right_knee→left_elbow |
| 171 | ang_embedders | `ang_31` | Signed angle (degrees) between vector left_hip→left_elbow and left_knee→left_elbow |
| 172 | gait_phases | `gait_00` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,20:22] trajectory norms over the gait cycle |
| 173 | gait_phases | `gait_01` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,16:18] trajectory norms over the gait cycle |
| 174 | gait_phases | `gait_02` | Normalized cross-correlation (×100) between embed_coor[:,32:34] and embed_coor[:,30:32] trajectory norms over the gait cycle |
| 175 | gait_phases | `gait_03` | Normalized cross-correlation (×100) between embed_coor[:,14:16] and embed_coor[:,12:14] trajectory norms over the gait cycle |
| 176 | gait_phases | `gait_04` | Normalized cross-correlation (×100) between embed_coor[:,28:30] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 177 | gait_phases | `gait_05` | Normalized cross-correlation (×100) between embed_coor[:,26:28] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 178 | gait_phases | `gait_06` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,30:32] trajectory norms over the gait cycle |
| 179 | gait_phases | `gait_07` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,32:34] trajectory norms over the gait cycle |
| 180 | gait_phases | `gait_08` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,30:32] trajectory norms over the gait cycle |
| 181 | gait_phases | `gait_09` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,32:34] trajectory norms over the gait cycle |
| 182 | gait_phases | `gait_10` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,16:18] trajectory norms over the gait cycle |
| 183 | gait_phases | `gait_11` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,18:20] trajectory norms over the gait cycle |
| 184 | gait_phases | `gait_12` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,16:18] trajectory norms over the gait cycle |
| 185 | gait_phases | `gait_13` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,18:20] trajectory norms over the gait cycle |
| 186 | gait_phases | `gait_14` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,12:14] trajectory norms over the gait cycle |
| 187 | gait_phases | `gait_15` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,14:16] trajectory norms over the gait cycle |
| 188 | gait_phases | `gait_16` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,12:14] trajectory norms over the gait cycle |
| 189 | gait_phases | `gait_17` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,14:16] trajectory norms over the gait cycle |
| 190 | gait_phases | `gait_18` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 191 | gait_phases | `gait_19` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 192 | gait_phases | `gait_20` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 193 | gait_phases | `gait_21` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 194 | gait_phases | `gait_22` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 195 | gait_phases | `gait_23` | Normalized cross-correlation (×100) between embed_coor[:,22:24] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 196 | gait_phases | `gait_24` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 197 | gait_phases | `gait_25` | Normalized cross-correlation (×100) between embed_coor[:,20:22] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 198 | gait_phases | `gait_26` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,30:32] trajectory norms over the gait cycle |
| 199 | gait_phases | `gait_27` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,32:34] trajectory norms over the gait cycle |
| 200 | gait_phases | `gait_28` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,30:32] trajectory norms over the gait cycle |
| 201 | gait_phases | `gait_29` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,32:34] trajectory norms over the gait cycle |
| 202 | gait_phases | `gait_30` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,12:14] trajectory norms over the gait cycle |
| 203 | gait_phases | `gait_31` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,14:16] trajectory norms over the gait cycle |
| 204 | gait_phases | `gait_32` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,12:14] trajectory norms over the gait cycle |
| 205 | gait_phases | `gait_33` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,14:16] trajectory norms over the gait cycle |
| 206 | gait_phases | `gait_34` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 207 | gait_phases | `gait_35` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 208 | gait_phases | `gait_36` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 209 | gait_phases | `gait_37` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 210 | gait_phases | `gait_38` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 211 | gait_phases | `gait_39` | Normalized cross-correlation (×100) between embed_coor[:,18:20] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 212 | gait_phases | `gait_40` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 213 | gait_phases | `gait_41` | Normalized cross-correlation (×100) between embed_coor[:,16:18] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 214 | gait_phases | `gait_42` | Normalized cross-correlation (×100) between embed_coor[:,32:34] and embed_coor[:,12:14] trajectory norms over the gait cycle |
| 215 | gait_phases | `gait_43` | Normalized cross-correlation (×100) between embed_coor[:,32:34] and embed_coor[:,14:16] trajectory norms over the gait cycle |
| 216 | gait_phases | `gait_44` | Normalized cross-correlation (×100) between embed_coor[:,30:32] and embed_coor[:,12:14] trajectory norms over the gait cycle |
| 217 | gait_phases | `gait_45` | Normalized cross-correlation (×100) between embed_coor[:,30:32] and embed_coor[:,14:16] trajectory norms over the gait cycle |
| 218 | gait_phases | `gait_46` | Normalized cross-correlation (×100) between embed_coor[:,32:34] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 219 | gait_phases | `gait_47` | Normalized cross-correlation (×100) between embed_coor[:,32:34] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 220 | gait_phases | `gait_48` | Normalized cross-correlation (×100) between embed_coor[:,30:32] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 221 | gait_phases | `gait_49` | Normalized cross-correlation (×100) between embed_coor[:,30:32] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 222 | gait_phases | `gait_50` | Normalized cross-correlation (×100) between embed_coor[:,32:34] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 223 | gait_phases | `gait_51` | Normalized cross-correlation (×100) between embed_coor[:,32:34] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 224 | gait_phases | `gait_52` | Normalized cross-correlation (×100) between embed_coor[:,30:32] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 225 | gait_phases | `gait_53` | Normalized cross-correlation (×100) between embed_coor[:,30:32] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 226 | gait_phases | `gait_54` | Normalized cross-correlation (×100) between embed_coor[:,14:16] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 227 | gait_phases | `gait_55` | Normalized cross-correlation (×100) between embed_coor[:,14:16] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 228 | gait_phases | `gait_56` | Normalized cross-correlation (×100) between embed_coor[:,12:14] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 229 | gait_phases | `gait_57` | Normalized cross-correlation (×100) between embed_coor[:,12:14] and embed_coor[:,28:30] trajectory norms over the gait cycle |
| 230 | gait_phases | `gait_58` | Normalized cross-correlation (×100) between embed_coor[:,14:16] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 231 | gait_phases | `gait_59` | Normalized cross-correlation (×100) between embed_coor[:,14:16] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 232 | gait_phases | `gait_60` | Normalized cross-correlation (×100) between embed_coor[:,12:14] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 233 | gait_phases | `gait_61` | Normalized cross-correlation (×100) between embed_coor[:,12:14] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 234 | gait_phases | `gait_62` | Normalized cross-correlation (×100) between embed_coor[:,28:30] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 235 | gait_phases | `gait_63` | Normalized cross-correlation (×100) between embed_coor[:,28:30] and embed_coor[:,26:28] trajectory norms over the gait cycle |
| 236 | gait_phases | `gait_64` | Normalized cross-correlation (×100) between embed_coor[:,26:28] and embed_coor[:,24:26] trajectory norms over the gait cycle |
| 237 | gait_phases | `gait_65` | Normalized cross-correlation (×100) between embed_coor[:,26:28] and embed_coor[:,26:28] trajectory norms over the gait cycle |

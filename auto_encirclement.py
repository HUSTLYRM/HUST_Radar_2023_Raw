import numpy as np

Encircle_Trigger_num = 1
Encircle_Activate_num = 1
blood_thres = 100

car2index = {
    1: 0,
    3: 1,
    4: 2,
    5: 3,
    7: 4,

    101: 0,
    103: 1,
    104: 2,
    105: 3,
    107: 4
}


# 传入参数：每辆车能够看到的装甲板类型
def Auto_encirclement(visible_list_total, blood, ENEMY_COLOR):
    visible_enemy = np.sum(visible_list_total, axis=0)

    # 如果超过3个机器人同时可见同一个目标, 开始围杀
    if np.max(visible_enemy) >= Encircle_Activate_num:
        maximum = np.max(visible_enemy)
        max_enemy = np.where(visible_enemy == maximum, 1, 0)

        if ENEMY_COLOR == 1:
            enemy1 = blood[0]
            enemy2 = blood[1]
            enemy3 = blood[2]
            enemy4 = blood[3]
            enemy5 = blood[4]
            enemywatch = blood[5]
            enemypost = blood[6]
            enemybase = blood[7]

        elif ENEMY_COLOR == 2:
            enemy1 = blood[8]
            enemy2 = blood[9]
            enemy3 = blood[10]
            enemy4 = blood[11]
            enemy5 = blood[12]
            enemywatch = blood[13]
            enemypost = blood[14]
            enemybase = blood[15]

        # 若血量为0, 不进行围杀
        for index in [0, 2, 3, 4]:
            if blood[index] == 0:
                visible_enemy[index] = 0
                max_enemy[index] = 0

        if enemywatch == 0:
            visible_enemy[6] = 0
            max_enemy[6] = 0

        # -1:英雄, -3/-4/-5:步兵, -7:哨兵
        # 哨兵最高优先级
        if max_enemy[-7] == 1:
            target = 6

        # 低于100血的英雄第二优先级
        elif max_enemy[-1] == 1 and enemy1 < blood_thres:
            target = 0

        # 低于blood_thres/可见度高的步兵
        elif (max_enemy[-3] + max_enemy[-4] + max_enemy[-5] >= 2) and min(enemy3, enemy4, enemy5) < blood_thres and \
                max_enemy[np.argmin(np.array(enemy3, enemy4, enemy5)) - 3]:

            target = np.argmin(np.array([enemy3, enemy4, enemy5])) - 3

        # 正常血量英雄
        elif max_enemy[-1] == 1:
            target = 0

        # 工程
        elif max_enemy[-2] == 1:
            target = 1
        # 空中机器人、8号忽略
        elif max_enemy[-6] == 1 or max_enemy[0] == 1:
            target = -1

        # 其余情况(步兵)按照序号+可见度选择(3,4,5)
        else:
            target = np.argmax(visible_enemy)

        return target

    else:
        return -1

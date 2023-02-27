from sklearn.metrics import f1_score

# all matched rules used for training data
y_00 = np.matmul(tr_0_ary, newc)
y_01 = np.matmul(tr_1_ary, newc)

pred0 = np.zeros(y_00.shape[0] + y_01.shape[0])
cnt0 = 0
cnt1 = 0
for i in range(y_00.shape[0]):

    cnt = 0
    for j in range(y_00.shape[1]):
        if (indc[j] == 1 and y_00[i, j] >= item_len[j]):
            pred0[i] = pred0[i] + conf[j]
            cnt = cnt + 1

    if (cnt == 0):
        cnt1 = cnt1 + 1
    else:
        pred0[i] = pred0[i] / cnt

cnt0 = 0
cnt1 = 0
for i in range(y_01.shape[0]):

    cnt = 0
    for j in range(y_01.shape[1]):
        if (indc[j] == 1 and y_01[i, j] >= item_len[j]):
            pred0[i] = pred0[i] + conf[j]
            cnt = cnt + 1

    if (cnt == 0):
        cnt1 = cnt1 + 1
    else:
        pred0[i + y_00.shape[0]] = pred0[i] / cnt

y0 = np.concatenate((np.zeros(tr_0_ary.shape[0]), np.ones(tr_1_ary.shape[0])), axis=0)

z_1 = pred0[y0 == 1]
m = np.mean(z_1)
s = np.std(z_1)
th = m

pred_y = np.zeros(pred.shape[0], dtype=int)
for i in range(pred.shape[0]):
    if (pred[i, 1] >= th):
        pred_y[i] = 1

print(f1_score(y, pred_y))

TP = 0
FP = 0
FN = 0
TN = 0
for i in range(pred.shape[0]):
    if (y[i] == 1 and pred_y[i] == 1):
        TP = TP + 1
    elif (y[i] == 1 and pred_y[i] == 0):
        FN = FN + 1
    elif (y[i] == 0 and pred_y[i] == 1):
        FP = FP + 1
    else:
        TN = TN + 1

print('TP:', TP, 'FN:', FN)
print('FP:', FP, 'TN:', TN)
print('pre:', TP / (TP + FP), 'rec:', TP / (TP + FN))
print('f1:', (2 * TP) / (2 * TP + FP + FN))
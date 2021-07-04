import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


model_path = 'ready_models/sofa_combo'

model = tf.keras.models.load_model(filepath=model_path, compile=False)
model.compile(optimizer='adam',
              loss=jaccard_distance, )

IMG_HEIGHT, IMG_WIDTH = model.get_config()['layers'][0]['config']['batch_input_shape'][1:3]


def create_mask(pred_mask):
    pred_mask = pred_mask[:, :, :, 0]
    pred_mask = tf.round(pred_mask)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def get_mask(img):
    # test_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    test_img = img.copy()
    color = 200

    re_img = cv2.resize(test_img.copy(),
                        dsize=(IMG_HEIGHT, IMG_WIDTH),
                        interpolation=cv2.INTER_CUBIC)

    coef_y = test_img.shape[0] / re_img.shape[0]
    coef_x = test_img.shape[1] / re_img.shape[1]

    pred = re_img / 255.
    pred = create_mask(model.predict(pred[tf.newaxis, ...]))
    pred = pred[:, :, 0].numpy()
    pred = np.stack((pred,) * 3, axis=-1)
    pred = np.uint8(pred * 255)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour[:, :, 0] = contour[:, :, 0] * coef_x
        contour[:, :, 1] = contour[:, :, 1] * coef_y

    try:
        filled_img = cv2.drawContours(test_img, [max(contours, key=cv2.contourArea)], -1, 255, 1)
        filled_img = cv2.fillPoly(test_img.copy(), pts=[max(contours, key=cv2.contourArea)], color=color, )
        # filled_img = cv2.drawContours(test_img, contours, -1, 255, 1)
        # filled_img = cv2.fillPoly(test_img.copy(), pts=contours, color=color,)
    except:
        return test_img

        # filled_img = cv2.drawContours(test_img, contours, -1, 255, 1)
        # filled_img = cv2.fillPoly(test_img.copy(), pts=contours, color=color,)

    cv2.addWeighted(test_img, 0.5, filled_img, 0.5, 0, test_img)

    box = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cv2.rectangle(test_img, box, color=color, thickness=4)
    return test_img


print(IMG_HEIGHT, IMG_WIDTH)

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
_, frame = cap.read()

while (True):
    _, frame = cap.read()
    frame = get_mask(frame)

    # frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

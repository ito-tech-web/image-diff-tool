import os
import cv2

#画像を読み込む。
def load_image(path): 
    """画像を読み込む。読めなければエラーにする。"""
    image = cv2.imread(path) #画像ファイルを読み込んで数値の配列にする
    if image is None:
        raise FileNotFoundError(f"画像が読み込めません: {path}") #ファイルがない場合にエラーにする。
    return image

#幅をそろえる
def resize(image, width=1000):
    """縦横比を保ったまま、幅を揃える。"""
    h, w = image.shape[:2] #幅と高さ
    scale = width / w #高さにも同じ倍率をかけて縦横比が崩れないように
    # 縮小するときは INTER_AREA を使うと画質がきれいに保てる。
    return cv2.resize(image, (width, int(h * scale)), interpolation=cv2.INTER_AREA)

#画像を比較しやすい形に整える
def preprocess(image):
    """グレースケール化 + ぼかしでノイズを減らす。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #カラーからグレースケールに変換する(OpenCVは色をRGBではなくBGRの順で持つため、BGR2GRAYと書く)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #ぼかす範囲の大きさと広がり具合を決める
    return blurred

#差分を求める
def make_diff_mask(good_gray, test_gray, threshold=30): 
    """2枚の差分を計算し、差が大きい部分を白にしたマスクを作る。"""
    diff = cv2.absdiff(good_gray, test_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY) #差が30より大きい画素は255(白),それ以外は0(黒)にして白黒の2画像にする
   #しきい値は使わないから捨てる変数名として、_を使用
    return diff, mask

#差があった場所を赤枠で囲む
def draw_boxes(image, mask, min_area=50):
    """差分があった場所を赤い四角で囲む。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 白い塊(輪郭)を探す
    result = image.copy()  # 元の画像を汚さないようにコピーして描く
    count = 0
    for cnt in contours:
        # 小さすぎる塊はノイズとして無視する
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 赤色(BGR)
        count += 1
    return result, count

#全体をつなぐ
def main():
    good = resize(load_image("images/good.jpg"))
    test = resize(load_image("images/test.jpg"))

    # 2枚のサイズが少しでも違うと差分計算でエラーになるため、testをgoodに合わせる
    test = cv2.resize(test, (good.shape[1], good.shape[0]))

    good_gray = preprocess(good)
    test_gray = preprocess(test)

    diff, mask = make_diff_mask(good_gray, test_gray)
    result, count = draw_boxes(test, mask)

    os.makedirs("output", exist_ok=True) # 保存先フォルダがなければ作る

    #画像をファイルに保存する
    cv2.imwrite("output/diff.jpg", diff)
    cv2.imwrite("output/mask.jpg", mask)
    cv2.imwrite("output/result.jpg", result)
    print(f"保存しました。差異のあった箇所: {count}件")


if __name__ == "__main__":
    main()
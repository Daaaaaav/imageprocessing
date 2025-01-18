import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageEnhance, ImageOps
import io
from scipy.signal import convolve2d
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def encode_image(image: Image.Image) -> str:
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    return base64.b64encode(img_buffer.getvalue()).decode()

def adjust_rgb(image, red_factor=1.0, green_factor=1.0, blue_factor=1.0):
    np_img = np.array(image)
    r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
    r = np.clip(r * red_factor, 0, 255)
    g = np.clip(g * green_factor, 0, 255)
    b = np.clip(b * blue_factor, 0, 255)
    adjusted_img = np.stack([r, g, b], axis=-1)
    return Image.fromarray(adjusted_img.astype('uint8'))

def apply_filter(image, filter_type):
    if filter_type == 'sepia':
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        np_img = np.array(image)
        filtered_img = np.dot(np_img[..., :3], sepia_filter.T)
        filtered_img = np.clip(filtered_img, 0, 255)
        return Image.fromarray(filtered_img.astype('uint8'))
    
def apply_custom_filter(image, color_wheel):
    np_img = np.array(image)
    height, width, _ = np_img.shape
    for y in range(height):
        for x in range(width):
            r, g, b = np_img[y, x]
            filter_color = color_wheel[(y + x) % len(color_wheel)]
            np_img[y, x] = [min(255, r * filter_color[0]), min(255, g * filter_color[1]), min(255, b * filter_color[2])]
    return Image.fromarray(np_img)

color_wheel = [(1.2, 0.8, 0.9), (0.9, 1.1, 0.8), (0.8, 0.9, 1.2)]

def histogram_equalization(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return Image.fromarray(equalized)

def contrast_stretching(image, lower_percentile=2, upper_percentile=98):
    np_img = np.array(image)
    min_val, max_val = np.percentile(np_img, (lower_percentile, upper_percentile))
    stretched = (np_img - min_val) * (255 / (max_val - min_val))
    stretched = np.clip(stretched, 0, 255)
    return Image.fromarray(stretched.astype('uint8'))

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return Image.fromarray(cv2.LUT(np.array(image), table))

def rle_compression(image):
    np_img = np.array(image).flatten()
    compressed = []
    count = 1
    for i in range(1, len(np_img)):
        if np_img[i] == np_img[i-1]:
            count += 1
        else:
            compressed.append((np_img[i-1], count))
            count = 1
    compressed.append((np_img[-1], count))
    return compressed

def dct_compression(image, quality=50):
    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(np.float32(np_img) / 255.0)
    compressed = dct[:quality, :quality]
    return Image.fromarray((compressed * 255).astype('uint8'))

def thresholding(image, method='global', thresh_value=128):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    if method == 'global':
        _, result = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(result)

def kmeans_segmentation(image, k=4):
    np_img = np.float32(np.array(image).reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(np_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()].reshape(image.size[1], image.size[0], 3)
    return Image.fromarray(segmented_img)

def morphological_operations(image, operation='dilation', kernel_size=5):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'dilation':
        result = cv2.dilate(gray, kernel)
    elif operation == 'erosion':
        result = cv2.erode(gray, kernel)
    elif operation == 'opening':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(result)

def boundary_extraction(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    eroded = cv2.erode(gray, np.ones((3, 3), np.uint8))
    boundary = gray - eroded
    return Image.fromarray(boundary)

def noise_reduction(image, method='gaussian'):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    if method == 'gaussian':
        result = cv2.GaussianBlur(gray, (5, 5), 0)
    elif method == 'wiener':
       def wiener_filter(noisy_image, kernel_size=5, noise_var=25, signal_var=100):  
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size) 
        blurred_image = convolve2d(noisy_image, kernel, mode='same', boundary='symm') 
        noise_est = noise_var 
        signal_est = signal_var 
        ratio = signal_est / (signal_est + noise_est) 
        filtered_image = blurred_image * ratio 
        return np.clip(filtered_image, 0, 255).astype(np.uint8) 
    result = wiener_filter(gray)
    return Image.fromarray(result)    

def inpainting(image, mask_path):
    mask = cv2.imread(mask_path, 0)
    result = cv2.inpaint(np.array(image), mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(result)

def feature_detection(image, method='sift'):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    if method == 'sift':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    elif method == 'orb':
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
    result = cv2.drawKeypoints(gray, keypoints, None)
    return Image.fromarray(result)

def template_matching(image, template_path):
    template = cv2.imread(template_path, 0)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = template.shape
    matched = cv2.rectangle(gray.copy(), top_left, (top_left[0] + w, top_left[1] + h), 255, 2)
    return Image.fromarray(matched)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    operation = request.form.get('operation', '').lower()
    value = request.form.get('value', '50')
    try:
        value = int(value)
    except ValueError:
            return jsonify({"error": "Invalid value input"}), 40

    image_file = request.files['image']
    operation = request.form.get('operation', '')
    value = int(request.form.get('value', 50))
    direction = request.form.get('direction', 'horizontal')
    
    image = Image.open(image_file)
    np_image = np.array(image)
    processed_image = image

    if operation == 'greyscale':
        processed_image = image.convert('L')
    elif operation == 'negative':
        processed_image = ImageOps.invert(image)
    elif operation == 'flip':
        if direction == 'horizontal':
            processed_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == 'vertical':
            processed_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            processed_image = image.transpose(Image.ROTATE_180)
    elif operation == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        processed_image = enhancer.enhance(value / 50)
    elif operation == 'contrast':
        enhancer = ImageEnhance.Contrast(image)
        processed_image = enhancer.enhance(value / 50)
    elif operation == 'rotate':
        processed_image = image.rotate(value)
    elif operation == 'scale':
        width, height = image.size
        processed_image = image.resize((int(width * value / 100), int(height * value / 100)))
    elif operation == 'translation':
        translated = np.roll(np_image, shift=value, axis=1) 
        processed_image = Image.fromarray(translated)
    elif operation == 'rgb_adjust':
        red = float(request.form.get('red_factor', 1.0))
        green = float(request.form.get('green_factor', 1.0))
        blue = float(request.form.get('blue_factor', 1.0))
        processed_image = adjust_rgb(image, red, green, blue)
    elif operation == 'color_filter':
        filter_type = request.form.get('filter_type', 'sepia')
        processed_image = apply_filter(image, filter_type)
    elif operation == 'crop':
        x = int(request.form.get('x', 0))
        y = int(request.form.get('y', 0))
        width = int(request.form.get('width', image.width))
        height = int(request.form.get('height', image.height))
        processed_image = image.crop((x, y, x + width, y + height))
    elif operation == 'blending':
        blend_image_file = request.files.get('blend_image')
        alpha = float(request.form.get('alpha', 0.5))
        blend_image = Image.open(blend_image_file)
        processed_image = Image.blend(image, blend_image, alpha)
    elif operation == 'add_border':
        border_size = int(request.form.get('border_size', 10))
        color = request.form.get('color', '#000000')
        processed_image = ImageOps.expand(image, border=border_size, fill=color)
    elif operation == 'add':
        value = int(request.form.get('value', 10))
        np_img = np_image + value
        np_img = np.clip(np_img, 0, 255)
        processed_image = Image.fromarray(np_img)
    elif operation == 'bitwise_and':
        mask_file = request.files.get('mask')
        mask = np.array(Image.open(mask_file))
        result = cv2.bitwise_and(np_image, mask)
        processed_image = Image.fromarray(result)
    elif operation == 'fft':
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        processed_image = Image.fromarray(magnitude_spectrum.astype('uint8'))
    elif operation == 'sobel':
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobel_x, sobel_y)
        processed_image = Image.fromarray(np.clip(sobel, 0, 255).astype('uint8'))
    elif operation == 'histogram_equalization':
        image = histogram_equalization(image)
    elif operation == 'gamma_correction':
        gamma_value = float(request.form['value'])
        image = gamma_correction(image, gamma=gamma_value)
    elif operation == 'rle_compression':
        compressed = rle_compression(image)
        image = Image.fromarray(np.uint8(compressed)) 


    buffer = io.BytesIO()
    processed_image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')

    return jsonify({'processed_image': encoded_image}), 200


if __name__ == '__main__':
    app.run(debug=True)

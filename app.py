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

def greyscale(image):
    return image.convert('L')

def negative(image):
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        inverted_image = ImageOps.invert(rgb_image)
        r, g, b = inverted_image.split()
        return Image.merge('RGBA', (r, g, b, a))
    else:
        return ImageOps.invert(image)

def change_color(image, target_color, replacement_color):
    np_img = np.array(image)
    target_color = np.array(target_color, dtype=np.uint8)
    replacement_color = np.array(replacement_color, dtype=np.uint8)
    if np_img.shape[2] == 4:
        target_color = np.append(target_color, 255)
        replacement_color = np.append(replacement_color, 255)
    mask = np.all(np_img[..., :3] == target_color[:3], axis=-1)
    if not np.any(mask):
        return image
    np_img[mask] = replacement_color
    return Image.fromarray(np_img)

def flip(image, direction):
    if direction == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return image.transpose(Image.ROTATE_180)
    
def diagonal_flip(image):
    return image.transpose(Image.TRANSPOSE)

def translation(image, tx, ty):
    width, height = image.size
    translation_matrix = (1, 0, tx, 0, 1, ty)
    return image.transform((width, height), Image.AFFINE, translation_matrix, resample=Image.BICUBIC)

def scale(image, value):
    width, height = image.size
    return image.resize((int(width * value / 100), int(height * value / 100)))

def rotate(image, angle):
    return image.rotate(angle)

def crop_image(image, x, y, width, height):
    np_img = np.array(image)
    img_height, img_width = np_img.shape[:2]
    x = max(0, min(int(x), img_width))
    y = max(0, min(int(y), img_height))
    width = max(1, min(int(width), img_width - x))
    height = max(1, min(int(height), img_height - y))
    cropped_img = np_img[y:y+height, x:x+width]
    return Image.fromarray(cropped_img)

def blend_images(image1, image2, alpha):
    return Image.blend(image1, image2, alpha)

def brightness(image, value):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(value / 50)

def contrast(image, value):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(value / 50)

def add_border(image, border_size, border_color):
    border_color = tuple(int(border_color[i:i+2], 16) for i in (0, 2, 4))
    return ImageOps.expand(image, border=border_size, fill=border_color)

def apply_filter(image, filter_type):
    if filter_type == 'sepia':
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        np_img = np.array(image)
        filtered_img = np.dot(np_img[..., :3], sepia_filter.T)
        filtered_img = np.clip(filtered_img, 0, 255)
        return Image.fromarray(filtered_img.astype('uint8'))
    elif filter_type == 'cyanotype':
        cyanotype_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
        np_img = np.array(image)
        filtered_img = np.dot(np_img[..., :3], cyanotype_filter.T)
        filtered_img = np.clip(filtered_img, 0, 255)
        return Image.fromarray(filtered_img.astype('uint8'))
    else:
        return image
    
def custom_filter(image, kernel):
    np_img = np.array(image)
    kernel = np.array(kernel, dtype=np.float32)
    filtered_img = cv2.filter2D(np_img, -1, kernel)
    return Image.fromarray(filtered_img)

def overlay_images(base_image, overlay_image, position, transparency):
    base_image = base_image.convert("RGBA")
    overlay_image = overlay_image.convert("RGBA")
    overlay_image = overlay_image.resize(base_image.size)
    blended = Image.blend(base_image, overlay_image, transparency)
    return blended

def pixel_addition(image1, image2):
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    result = cv2.add(np_img1, np_img2)
    return Image.fromarray(result)

def pixel_subtraction(image1, image2):
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    result = cv2.subtract(np_img1, np_img2)
    return Image.fromarray(result)

def pixel_multiplication(image1, image2):
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    result = cv2.multiply(np_img1, np_img2)
    return Image.fromarray(result)

def pixel_division(image1, image2):
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    result = cv2.divide(np_img1, np_img2)
    return Image.fromarray(result)

def bitwise_and(image1, image2):
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    result = cv2.bitwise_and(np_img1, np_img2)
    return Image.fromarray(result)

def bitwise_or(image1, image2):
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    result = cv2.bitwise_or(np_img1, np_img2)
    return Image.fromarray(result)

def bitwise_xor(image1, image2):
    np_img1 = np.array(image1)
    np_img2 = np.array(image2)
    result = cv2.bitwise_xor(np_img1, np_img2)
    return Image.fromarray(result)

def bitwise_not(image):
    np_img = np.array(image)
    result = cv2.bitwise_not(np_img)
    return Image.fromarray(result)

def fourier_transform(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255)
    return Image.fromarray(magnitude_spectrum.astype('uint8'))

def mean_filter(image, ksize):
    return Image.fromarray(cv2.blur(np.array(image), (ksize, ksize)))

def gaussian_filter(image, ksize):
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number!")
    return Image.fromarray(cv2.GaussianBlur(np.array(image), (ksize, ksize), 0))

def median_filter(image, ksize):
    if ksize <= 0 or ksize % 2 == 0:
       raise ValueError("Kernel size must be a positive odd number!")
    return Image.fromarray(cv2.medianBlur(np.array(image), ksize))

def sobel_filter(image, dx, dy, ksize):
    sobel = cv2.Sobel(np.array(image), cv2.CV_64F, dx, dy, ksize)
    abs_sobel = np.absolute(sobel)
    sobel_8u = np.uint8(abs_sobel)
    return Image.fromarray(sobel_8u)

def canny_edge(image, threshold1, threshold2):
    return Image.fromarray(cv2.Canny(np.array(image), threshold1, threshold2))

def laplacian_filter(image):
    laplacian = cv2.Laplacian(np.array(image), cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    laplacian_8u = np.uint8(abs_laplacian)
    return Image.fromarray(laplacian_8u)

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

def global_thresholding(image, threshold):
    _, thresh_img = cv2.threshold(np.array(image), threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh_img)

def adaptive_thresholding(image, method, block_size, C):
    if block_size <= 0 or block_size % 2 == 0:
       raise ValueError("Block size must be a positive odd number!")
    if method == 'mean':
        return Image.fromarray(cv2.adaptiveThreshold(np.array(image), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C))
    elif method == 'gaussian':
        return Image.fromarray(cv2.adaptiveThreshold(np.array(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C))

def k_means_clustering(image, K):
    Z = np.array(image).reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return Image.fromarray(res.reshape((image.size[1], image.size[0], 3)))

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

def skeletonization(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary, open)
        eroded = cv2.erode(binary, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    return Image.fromarray(skeleton)

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

def inpaint_image(image, mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Mask image could not be loaded")
    np_img = np.array(image)
    if np_img.shape[:2] != mask.shape:
        raise ValueError("Mask dimensions do not match image dimensions")
    inpainted = cv2.inpaint(np_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)

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

def template_matching(image, template_image):
    template = np.array(template_image.convert('L'))
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
        raise ValueError("Template dimensions are larger than image dimensions")
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(gray, top_left, bottom_right, 255, 2)
    return Image.fromarray(gray)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info('Received request for image processing')
        image_file = request.files.get('image')
        if not image_file:
            raise ValueError("No image file provided")
        
        operation = request.form.get('operation')
        if not operation:
            raise ValueError("No operation specified")
        
        app.logger.info(f'Operation: {operation}')

        img = Image.open(image_file)
        value = request.form.get('value', '0')
        angle = request.form.get('angle', '0')
        direction = request.form.get('direction', '')
        gamma = request.form.get('gamma', '1')
        target_color = request.form.get('target_color', 'ff0000').lstrip('#')
        replacement_color = request.form.get('replacement_color', '00ff00').lstrip('#')
        tx = request.form.get('tx', '0')
        ty = request.form.get('ty', '0')
        left = request.form.get('left', '0')
        top = request.form.get('top', '0')
        right = request.form.get('right', '0')
        bottom = request.form.get('bottom', '0')
        alpha = request.form.get('alpha', '1')
        border_size = request.form.get('border_size', '10')
        border_color = request.form.get('border_color', '000000').lstrip('#')
        color = request.form.get('color', 'black')
        mask_path = request.form.get('mask_path', '')
        template_path = request.form.get('template_path', '')

        app.logger.info(f'Received values: value={value}, angle={angle}, direction={direction}, gamma={gamma}')
        app.logger.info(f'target_color={target_color}, replacement_color={replacement_color}')
        app.logger.info(f'tx={tx}, ty={ty}, left={left}, top={top}, right={right}, bottom={bottom}')
        app.logger.info(f'alpha={alpha}, border_size={border_size}, border_color={border_color}, color={color}')
        app.logger.info(f'mask_path={mask_path}, template_path={template_path}')

        if operation in ['brightness', 'contrast', 'scale', 'translation', 'mean_filter', 'gaussian_filter', 'median_filter', 'sobel_filter', 'canny_edge', 'global_thresholding', 'adaptive_thresholding', 'k_means_clustering', 'morphological']:
            value = int(value or '0')
        if operation == 'rotate':
            angle = int(angle or '0')
        if operation == 'gamma_correction':
            gamma = float(gamma or '1')
        if operation == 'change_color':
            target_color = tuple(int(target_color[i:i+2], 16) for i in (0, 2, 4))
            replacement_color = tuple(int(replacement_color[i:i+2], 16) for i in (0, 2, 4))
        if operation == 'translation':
            tx = int(tx or '0')
            ty = int(ty or '0')
        if operation == 'crop':
            x = int(request.form.get('cropX', 0))
            y = int(request.form.get('cropY', 0))
            width = int(request.form.get('cropWidth', img.width))
            height = int(request.form.get('cropHeight', img.height))
            processed_img = crop_image(img, x, y, width, height)
        if operation in ['blend', 'overlay', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'pixel_addition', 'pixel_subtraction', 'pixel_multiplication', 'pixel_division']:
            alpha = float(alpha or '1')
        if operation == 'add_border':
            border_size = int(border_size or '10')
            border_color = tuple(int(border_color[i:i+2], 16) for i in (0, 2, 4))

        app.logger.info('Processing image...')
        app.logger.info(f'Image size: {img.size}, mode: {img.mode}')
        
        if operation in ['blend', 'overlay', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'pixel_addition', 'pixel_subtraction', 'pixel_multiplication', 'pixel_division']:
            image_file2 = request.files.get('image2')
            if not image_file2:
                raise ValueError(f"No second image file provided for {operation}")
            img2 = Image.open(image_file2)
            app.logger.info(f'Second image size: {img2.size}, mode: {img2.mode}')
            if img.size != img2.size or img.mode != img2.mode:
                img2 = img2.resize(img.size)
                app.logger.info(f'Resized second image to: {img2.size}, mode: {img2.mode}')

        if operation == 'greyscale':
            processed_img = greyscale(img)
        elif operation == 'negative':
            processed_img = negative(img)
        elif operation == 'flip':
            if direction == 'horizontal':
                processed_img = flip(img, 'horizontal')
            elif direction == 'vertical':
                processed_img = flip(img, 'vertical')
            else:
                processed_img = diagonal_flip(img)
        elif operation == 'brightness':
            processed_img = brightness(img, value)
        elif operation == 'contrast':
            processed_img = contrast(img, value)
        elif operation == 'rotate':
            processed_img = rotate(img, angle)
        elif operation == 'scale':
            processed_img = scale(img, value)
        elif operation == 'translation':
            processed_img = translation(img, tx, ty)
        elif operation == 'crop':
            processed_img = crop_image(img, left, top, right, bottom)
        elif operation == 'blend':
            processed_img = blend_images(img, img2, alpha)
        elif operation == 'change_color':
            processed_img = change_color(img, target_color, replacement_color)
        elif operation == 'adjust_rgb':
            red_factor = float(request.form.get('red_factor', '1.0'))
            green_factor = float(request.form.get('green_factor', '1.0'))
            blue_factor = float(request.form.get('blue_factor', '1.0'))
            processed_img = adjust_rgb(img, red_factor=red_factor, green_factor=green_factor, blue_factor=blue_factor)
        elif operation == 'sepia_filter':
            processed_img = apply_filter(img, 'sepia')
        elif operation == 'cyano_filter':
            processed_img = apply_filter(img, 'cyanotype')
        elif operation == 'custom_filter':
            kernel = np.array(eval(request.form.get('kernel', '[[1,1,1],[1,1,1],[1,1,1]]')))
            processed_img = custom_filter(img, kernel)
        elif operation == 'fourier_transform':
            processed_img = fourier_transform(img)
        elif operation == 'mean_filter':
            processed_img = mean_filter(img, value)
        elif operation == 'gaussian_filter':
            processed_img = gaussian_filter(img, value)
        elif operation == 'median_filter':
            processed_img = median_filter(img, value)
        elif operation == 'sobel_filter':
            processed_img = sobel_filter(img, 1, 1, value)
        elif operation == 'canny_edge':
            processed_img = canny_edge(img, value, value * 2)
        elif operation == 'laplacian_filter':
            processed_img = laplacian_filter(img)
        elif operation == 'histogram_equalization':
            processed_img = histogram_equalization(img)
        elif operation == 'gamma_correction':
            processed_img = gamma_correction(img, gamma)
        elif operation == 'rle_compression':
            processed_img = rle_compression(img)
        elif operation == 'dct_compression':
            processed_img = dct_compression(img)
        elif operation == 'global_thresholding':
            processed_img = global_thresholding(img, value)
        elif operation == 'adaptive_thresholding':
            processed_img = adaptive_thresholding(img, 'mean', value, value)
        elif operation == 'k_means_clustering':
            processed_img = k_means_clustering(img, value)
        elif operation == 'morphological':
            processed_img = morphological_operations(img, 'dilation', value)
        elif operation == 'boundary_extraction':
            processed_img = boundary_extraction(img)
        elif operation == 'skeletonization':
            processed_img = skeletonization(img)
        elif operation == 'noise_reduction':
            processed_img = noise_reduction(img, 'gaussian')
        elif operation == 'inpainting':
            if not mask_path:
                raise ValueError("No mask path provided for inpainting")
            processed_img = inpaint_image(img, mask_path)
        elif operation == 'feature_detection':
            processed_img = feature_detection(img)
        elif operation == 'template_matching':
            if not template_path:
                raise ValueError("No template path provided for template matching")
            processed_img = template_matching(img, template_path)
        elif operation == 'overlay':
            processed_img = overlay_images(img, img2, (tx, ty), alpha)
        elif operation == 'add_border':
            processed_img = add_border(img, border_size, color)
        elif operation == 'bitwise_and':
            processed_img = bitwise_and(img, img2)
        elif operation == 'bitwise_or':
            processed_img = bitwise_or(img, img2)
        elif operation == 'bitwise_xor':
            processed_img = bitwise_xor(img, img2)
        elif operation == 'bitwise_not':
            processed_img = bitwise_not(img)
        elif operation == 'pixel_addition':
            processed_img = pixel_addition(img, img2)
        elif operation == 'pixel_subtraction':
            processed_img = pixel_subtraction(img, img2)
        elif operation == 'pixel_multiplication':
            processed_img = pixel_multiplication(img, img2)
        elif operation == 'pixel_division':
            processed_img = pixel_division(img, img2)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        app.logger.info('Image processed successfully')
        processed_image_base64 = encode_image(processed_img)
        return jsonify({'image': processed_image_base64})

    except Exception as e:
        app.logger.error(f'Error processing image: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
function toggleOptions() {
    const operation = document.getElementById('operation').value;
    const valueGroup = document.getElementById('valueGroup');
    const valueSlider = document.getElementById('value');
    const valueNumber = document.getElementById('valueNumber');
    const colorGroup = document.getElementById('colorGroup');
    const rgbGroup = document.getElementById('rgbGroup');
    const customFilterGroup = document.getElementById('customFilterGroup');
    const borderGroup = document.getElementById('borderGroup');
    const maskGroup = document.getElementById('maskGroup');
    const templateGroup = document.getElementById('templateGroup');
    const secondImageGroup = document.getElementById('secondImageGroup');
    const angleGroup = document.getElementById('angleGroup');
    const cropGroup = document.getElementById('cropGroup');

    const sliderOperations = ['brightness', 'contrast', 'scale', 'rotation', 'translation', 'mean_filter', 'sobel_filter', 'canny_edge', 'global_thresholding', 'adaptive_thresholding', 'k_means_clustering', 'morphological'];
    const numberOperations = ['gaussian_filter', 'median_filter'];
    const secondImageOperations = ['blend', 'overlay', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'pixel_addition', 'pixel_subtraction', 'pixel_multiplication', 'pixel_division', 'template_matching', 'inpainting'];
    const cropOperations = ['crop'];

    if (sliderOperations.includes(operation)) {
        valueGroup.style.display = 'block';
        valueSlider.style.display = 'block';
        valueNumber.style.display = 'none';
        angleGroup.style.display = operation === 'rotation' ? 'block' : 'none';
        cropGroup.style.display = 'none';
        maskGroup.style.display = 'none';
    } else if (numberOperations.includes(operation)) {
        valueGroup.style.display = 'block';
        valueSlider.style.display = 'none';
        valueNumber.style.display = 'block';
        angleGroup.style.display = 'none';
        cropGroup.style.display = 'none';
        maskGroup.style.display = 'none';
    } else if (cropOperations.includes(operation)) {
        valueGroup.style.display = 'none';
        angleGroup.style.display = 'none';
        cropGroup.style.display = 'block';
        maskGroup.style.display = 'none';
    } else if (operation === 'inpainting') {
        valueGroup.style.display = 'none';
        angleGroup.style.display = 'none';
        cropGroup.style.display = 'none';
        maskGroup.style.display = 'block';
    } else {
        valueGroup.style.display = 'none';
        angleGroup.style.display = 'none';
        cropGroup.style.display = 'none';
        maskGroup.style.display = 'none';
    }
    angleGroup.style.display = operation === 'rotate' ? 'block' : 'none';
    colorGroup.style.display = operation === 'change_color' ? 'block' : 'none';
    rgbGroup.style.display = operation === 'adjust_rgb' ? 'block' : 'none';
    customFilterGroup.style.display = operation === 'custom_filter' ? 'block' : 'none';
    borderGroup.style.display = operation === 'add_border' ? 'block' : 'none';
    maskGroup.style.display = operation === 'inpainting' ? 'block' : 'none';
    templateGroup.style.display = operation === 'template_matching' ? 'block' : 'none';
    secondImageGroup.style.display = secondImageOperations.includes(operation) ? 'block' : 'none';

    document.getElementById('angleGroup').style.display = operation === 'rotate' ? 'block' : 'none';
    document.getElementById('directionGroup').style.display = operation === 'flip' ? 'block' : 'none';
    document.getElementById('gammaGroup').style.display = operation === 'gamma_correction' ? 'block' : 'none';
    document.getElementById('translationGroup').style.display = operation === 'translation' ? 'block' : 'none';
    document.getElementById('cropGroup').style.display = operation === 'crop' ? 'block' : 'none';
    document.getElementById('blendGroup').style.display = ['blend', 'overlay', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'pixel_addition', 'pixel_subtraction', 'pixel_multiplication', 'pixel_division'].includes(operation) ? 'block' : 'none';
    document.getElementById('blendGroup').style.display = ['blend', 'overlay', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'pixel_addition', 'pixel_subtraction', 'pixel_multiplication', 'pixel_division'].includes(operation) ? 'block' : 'none';
}

document.getElementById('operation').addEventListener('change', toggleOptions);

async function uploadImage(event) {
    event.preventDefault();
    const form = document.getElementById('imageForm');
    const formData = new FormData(form);
    const operation = document.getElementById('operation').value;

    formData.append('operation', operation);

    if (operation === 'rotate') {
        formData.append('angle', document.getElementById('angle').value || '0');
    }

    if (operation === 'inpainting') {
        formData.append('image', document.getElementById('image').files[0]);
        formData.append('mask', document.getElementById('mask').files[0]);
    }

    if (operation === 'template_matching' || operation === 'inpainting') {
        const image2 = document.getElementById('image2').files[0];
        if (image2) {
            formData.append('image2', image2);
        }
    }

    if (operation === 'overlay') {
        formData.append('tx', document.getElementById('tx').value || '0');
        formData.append('ty', document.getElementById('ty').value || '0');
        formData.append('alpha', document.getElementById('alpha').value || '0.5');
    }

    const value = document.getElementById('value').style.display === 'block' ? document.getElementById('value').value : document.getElementById('valueNumber').value;
    formData.append('value', value);

    if (operation === 'change_color') {
        formData.append('target_color', document.getElementById('target_color').value.replace('#', ''));
        formData.append('replacement_color', document.getElementById('replacement_color').value.replace('#', ''));
    }
    if (operation === 'adjust_rgb') {
        formData.append('red_factor', document.getElementById('red_factor').value);
        formData.append('green_factor', document.getElementById('green_factor').value);
        formData.append('blue_factor', document.getElementById('blue_factor').value);
    }
    if (operation === 'custom_filter') {
        formData.append('kernel', document.getElementById('custom_filter_kernel').value);
    }
    if (operation === 'add_border') {
        formData.append('border_size', document.getElementById('border_size').value);
        formData.append('border_color', document.getElementById('border_color').value.replace('#', ''));
    }

    if (['blend', 'overlay', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'pixel_addition', 'pixel_subtraction', 'pixel_multiplication', 'pixel_division'].includes(operation)) {
        const image2 = document.getElementById('image2').files[0];
        if (image2) {
            formData.append('image2', image2);
        }
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            alert(`Error: ${errorData.error}`);
        } else {
            const resultData = await response.json();
            document.getElementById('resultImage').src = `data:image/png;base64,${resultData.image}`;
            document.getElementById('result').style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the image.');
    }
}

document.getElementById('imageForm').onsubmit = uploadImage;
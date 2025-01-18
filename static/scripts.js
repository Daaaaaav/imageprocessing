document.getElementById('operation').addEventListener('change', toggleOptions);

function toggleOptions() {
    const operation = document.getElementById('operation').value;
    const valueInput = document.getElementById('valueGroup');
    const flipGroup = document.getElementById('flipDirectionGroup');
    const rotationGroup = document.getElementById('rotationGroup');
    const gammaGroup = document.getElementById('gammaCorrectionGroup');
    const translationGroup = document.getElementById('translationGroup');
    valueInput.style.display = (['brightness', 'contrast', 'scale', 'translation'].includes(operation)) ? 'block' : 'none';
    translationGroup.style.display = operation === 'translation' ? 'block' : 'none';
    flipGroup.style.display = (operation === 'flip') ? 'block' : 'none';
    rotationGroup.style.display = (operation === 'rotate') ? 'block' : 'none';
    gammaGroup.style.display = (operation === 'gamma_correction') ? 'block' : 'none';
} 

document.getElementById('dxSlider').addEventListener('input', function() {
    document.getElementById('dxValue').textContent = this.value;
});

document.getElementById('dySlider').addEventListener('input', function() {
    document.getElementById('dyValue').textContent = this.value;
});


async function uploadImage() {
    const form = document.getElementById('imageForm');
    const formData = new FormData(form);
    const operation = document.getElementById('operation').value;
    
    if (['brightness', 'contrast', 'scale', 'translation'].includes(operation)) {
        formData.append('value', document.getElementById('value').value);
    }
    if (operation === 'rotate') {
        formData.append('value', document.getElementById('angle').value);
    }
    if (operation === 'flip') {
        formData.append('direction', document.getElementById('direction').value);
    }
    
    const dx = document.getElementById('dxSlider').value;
    const dy = document.getElementById('dySlider').value;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            document.getElementById('predictionText').textContent = errorData.error || 'Error processing the image';
        } else {
            const resultData = await response.json();
            document.getElementById('processedImage').src = 'data:image/png;base64,' + resultData.processed_image;
            document.getElementById('predictionText').textContent = 'Image processed successfully!';
        }
    } catch (error) {
        document.getElementById('predictionText').textContent = 'Error communicating with the server';
        console.error('Error:', error);
    }
}
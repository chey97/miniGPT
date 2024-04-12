function predict() {
    // Get input text from the textarea
    const inputText = document.getElementById('input-text').value;

    // Make a POST request to the Flask backend
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            input_data: inputText
        })
    })
    .then(response => response.json())
    .then(data => {
        // Update the output-text paragraph with the predicted answer
        document.getElementById('output-text').textContent = data.predicted_answer;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

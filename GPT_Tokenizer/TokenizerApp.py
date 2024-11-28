from flask import Flask, request, jsonify, render_template_string
from minimal import BasicTokenizer

app = Flask(__name__)

# HTML template for the index page
index_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Text Tokenizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f8ff;
            color: #333;
        }
        h1 {
            color: #4682b4;
        }
        textarea {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            box-sizing: border-box;
        }
        .token { display: inline-block; padding: 2px; margin: 1px; }
        .color-0 { background-color: #ffcccc; }
        .color-1 { background-color: #ccffcc; }
        .color-2 { background-color: #ccccff; }
        .color-3 { background-color: #ffffcc; }
        .color-4 { background-color: #ffccff; }
        .color-5 { background-color: #ccffff; }
    </style>
</head>
<body>
    <h1>Text Tokenizer</h1>
    <textarea id="text-input" name="text" rows="10" oninput="tokenizeText()"></textarea>
    <h2>Tokenized Text</h2>
    <div id="tokenized-output"></div>
    <script>
        function tokenizeText() {
            const text = document.getElementById('text-input').value;
            fetch('/tokenize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                const outputDiv = document.getElementById('tokenized-output');
                outputDiv.innerHTML = data.tokenized_text.map((token, index) => {
                    const colorClass = 'color-' + (index % 6); // Cycle through 6 colors
                    return `<span class="token ${colorClass}">${token}</span>`;
                }).join('');
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(index_html)

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.get_json()
    text = data['text']
    # Tokenize the text using your tokenizer
    tokenizer = BasicTokenizer()
    tokenized_text = tokenizer.encode(text)
    return jsonify({'tokenized_text': tokenized_text})

if __name__ == '__main__':
    app.run(debug=True)

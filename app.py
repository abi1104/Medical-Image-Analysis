from flask import Flask, request, render_template, jsonify
import os
from predict_scripts.predict_tumor import predict_tumor  
from predict_scripts.predict_pneumonia import predict_pneumonia  

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def upload_file():
    return render_template('upload.html') 
@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        
        file = request.files['file']
        image_type = request.form.get('image_type')  

        
        if file and file.filename.endswith(('.png', '.jpg', '.jpeg')) and image_type:
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

           
            if image_type == 'mri':
                predicted_class = predict_tumor(file_path)  
            #elif image_type == 'ct':
                #predicted_class = predict_lung_disease(file_path)
            elif image_type == 'xray':
                predicted_class = predict_pneumonia(file_path) 
            else:
                return jsonify({'error': 'Invalid image type selected.'}), 400
            return jsonify({'prediction': predicted_class})

        else:
            return jsonify({'error': 'Invalid file format or image type not selected.'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

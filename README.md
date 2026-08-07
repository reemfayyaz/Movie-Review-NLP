# 🤖 Movie Review NLP - Text Classification App

A machine learning-powered text classification application built with **Streamlit** that predicts the category of movie reviews and other text inputs using a trained NLP model.

## 📋 Features

- **Text Classification**: Automatically classify text into predefined categories
- **Confidence Scores**: View probability predictions for each class
- **User-Friendly Interface**: Clean and intuitive web application built with Streamlit
- **Model Status Monitoring**: Real-time model loading status and information
- **Session State Management**: Persistent prediction history within a session
- **Error Handling**: Comprehensive error messages and technical details

## 🛠️ Tech Stack

- **Python 3.x**
- **Streamlit** - Web app framework
- **scikit-learn** - Machine learning library
- **joblib** - Model serialization
- **Pandas** - Data manipulation
- **pathlib** - File path handling

## 📁 Project Structure

```
Movie-Review-NLP/
├── app(7).py                 # Main Streamlit application
├── model.pkl                 # Trained classification model
├── vectorizer.pkl            # Text vectorizer (TfidfVectorizer/CountVectorizer)
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies (optional)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/reemfayyaz/Movie-Review-NLP.git
   cd Movie-Review-NLP
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit scikit-learn joblib pandas
   ```

### Running the Application

```bash
streamlit run app(7).py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 How to Use

1. **Enter Text**: Paste or type the text you want to classify in the input area
2. **Analyze**: Click the "🔍 Analyze Text" button to get predictions
3. **Review Results**: View the predicted category and confidence scores
4. **Clear**: Click the "🗑️ Clear" button to reset the form

## 🔧 Model Details

### Vectorizer
- **Type**: TfidfVectorizer or CountVectorizer
- **File**: `vectorizer.pkl`
- **Purpose**: Converts text input into numerical features for the model

### Classification Model
- **File**: `model.pkl`
- **Features**:
  - Supports `predict()` for class predictions
  - Supports `predict_proba()` for confidence scores (if available)
  - Includes `classes_` attribute for class labels

## 📊 Application Layout

### Left Column (Input)
- Text area for user input
- Analyze and Clear buttons
- Instructions and guidelines

### Right Column (Results)
- Predicted category display
- Confidence scores for each class
- Progress bars for visual representation
- Real-time feedback

### Sidebar
- Model status indicator
- How to use instructions
- Model information (type, classes, vectorizer)
- Author attribution

## ⚠️ Troubleshooting

### Model Files Not Found
**Error**: "model.pkl was not found" or "vectorizer.pkl was not found"

**Solution**: 
- Ensure both `model.pkl` and `vectorizer.pkl` are in the same directory as `app(7).py`
- Check file names for exact spelling

### Application Won't Start
**Error**: Port already in use

**Solution**:
```bash
streamlit run app(7).py --server.port 8502
```

### Prediction Errors
**Error**: "An error occurred during prediction"

**Solution**:
- Verify model files are not corrupted
- Check that vectorizer and model are compatible
- Review technical error details in the expandable section

## 🔄 Caching

The application uses Streamlit's `@st.cache_resource` decorator to cache the model and vectorizer, ensuring:
- Fast load times after the first run
- Efficient resource usage
- Consistent predictions across sessions

## 📝 Session Management

The app maintains session state for:
- **prediction**: Last predicted category
- **probabilities**: Confidence scores for each class
- **input_text**: Current text in the input area

This allows users to navigate and review results without losing information.

## 🎨 Styling

The application includes custom CSS styling for:
- Responsive layout (wide configuration)
- Professional color scheme
- Rounded buttons and containers
- Clean typography
- Optimal spacing and padding

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

REEM FAYYAZ

## 🙏 Acknowledgments

- Streamlit team for the amazing web app framework
- scikit-learn community for excellent ML tools
- All contributors and users of this project

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Happy Classifying! 🚀**

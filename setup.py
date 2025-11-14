import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ia_que_cuida",
    version="0.0.1",
    author="",
    author_email="",
    description="Cuidado de personas mayores con IA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[],
    python_requires=">=3.6",
    install_requires=[
        "streamlit>=0.63",
        "pydub>=0.24",
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "altair",
        "pyttsx3",
        "whisper",
        "torch",
        "openai",
        "emoji",
        "joblib",
        "scikit-learn",
        "ipython",
        "google-cloud-texttospeech",
        "playsound==1.2.2",
        "pysentimiento",
        "openai-whisper"

        #"transformers==4.3.3"

    ],
)

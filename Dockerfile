FROM tensorflow/tensorflow:1.14.0-gpu-py3

COPY . /Workspace

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -U scikit-learn

WORKDIR /Workspace

CMD python main_UCI_idg.py && \
    more /Workspace/Experiment_logs/UCI_HAR_Feature_Experiment.txt
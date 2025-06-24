FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY torque.py ${LAMBDA_TASK_ROOT}

# Copy data required
ENV TORQUE=/var/torque/
RUN mkdir -p ${TORQUE}
COPY diagram.jpg ${TORQUE}
COPY easyocrmodel ${TORQUE}/easyocrmodel

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]

import logging

# Create a logger
logger = logging.getLogger('PERSONALIZED')
logger.setLevel(logging.DEBUG)  # Set the log level for the logger

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)  # Set the log level for the handler

# Create a console handler to output log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the log level for the handler

# Define the log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

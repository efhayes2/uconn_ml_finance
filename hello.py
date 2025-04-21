# Define a simple function that prints a greeting
def say_hello():
  """Prints a greeting message."""
  print("Hello from the say_hello function!")
  print("This is part of the uconn_ml_finance project.")

# The standard Python entry point check
# Code inside this block runs ONLY when the script is executed directly
if __name__ == "__main__":
  # This message will print when you run the script directly
  print("Executing hello.py directly...")

  # Call the function defined above
  say_hello()

  # This message will print after the function call
  print("Script finished.")

# If this script were imported into another file,
# the code inside the 'if __name__ == "__main__":' block would NOT run.
# Only the function definition 'say_hello' would be available for use.

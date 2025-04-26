# This script formats the user's persona information
def format_persona(user_data):
    return f"User Persona: {user_data}"

if __name__ == "__main__":
    user_data = {"name": "John Doe", "interests": ["AI", "Data Science"]}
    print(format_persona(user_data))
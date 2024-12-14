from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
import joblib


# Load the trained model
model = joblib.load('tintor.pkl')

# Define constants
NUM_MENTORS = 300
SKILLS = ["Python", "Leadership", "Time Management", "Data Analysis", "Project Management",
          "Java", "Marketing", "Public Speaking", "Teamwork", "Creative Thinking", "UI/UX Design"]
DEPARTMENTS = ["DSG", "LSG", "MOC", "TSG", "OOC", "QCC", "EDS", "SSG", "POC", "CLV", "OM Detention & Demurrage Management", "HR"]
INTERESTS = ["Sports", "Reading", "Traveling", "Cooking"]

# Function to generate a random Vietnamese phone number
def generate_phone_number():
    prefixes = ["09", "05", "03"]
    return random.choice(prefixes) + ''.join([str(random.randint(0, 9)) for _ in range(8)])

# Helper function to generate Vietnamese names
def generate_vietnamese_name():
    first_names = ["Anh", "Binh", "Cuong", "Dung", "Hanh", "Huy", "Linh", "Minh", "Ngan", "Phat", "Quang", "Tu", "Trang"]
    last_names = ["Nguyen", "Tran", "Le", "Pham", "Huynh", "Hoang", "Dang", "Bui", "Vu", "Do"]
    return f"{random.choice(last_names)} {random.choice(first_names)}"

# Define the specific mentor names and data
mentor_skills_and_departments = {
    "Khoa Nguyen": {"skills": ["Python", "Leadership", "Time Management", "Data Analysis"], "Department": "SSG"},
    "Dat Tran": {"skills": ["Java", "Teamwork", "Project Management"], "Department": "CLV"},
    "Thi Huynh": {"skills": ["Creative Thinking", "Marketing", "Public Speaking", "Teamwork"], "Department": "OM Detention & Demurrage Management"},
    "My Nguyen": {"skills": ["Project Management", "Public Speaking", "Teamwork", "Creative Thinking"], "Department": "OM Detention & Demurrage Management"},
    "Phu Nguyen": {"skills": ["UI/UX Design", "Time Management"], "Department": "OM Detention & Demurrage Management"},
    "Chau Tran": {"skills": ["Creative Thinking", "Marketing", "Public Speaking", "Teamwork"], "Department": "OM Detention & Demurrage Management"},
    "Phat Vo": {"skills": ["Java", "Teamwork", "Project Management"], "Department": "CLV"}
}

# Generate mentor data for the fixed mentors
mentor_data = [
    {
        "mentorID": f"M{i+1:02d}",
        "name": name,
        "skills": details["skills"],
        "experience": random.randint(10, 30),
        "department": details["Department"],
        "mentoringCapacity": random.randint(1, 5),
        "preferredSkills": random.sample(SKILLS, k=random.randint(2, 4)),
        "availability": random.choice(["Flexible", "Specific hours"]),
        "email": f"{name.split(' ')[-1].lower()}.{name.split(' ')[0].lower()}@cyberlogitec.com",
        "interest": random.choice(INTERESTS),
        "phoneNumber": generate_phone_number()
    }
    for i, (name, details) in enumerate(mentor_skills_and_departments.items())
]

# Generate additional random mentor data for the remaining mentors (293)
additional_mentors_count = NUM_MENTORS - len(mentor_skills_and_departments)

# Generate additional mentor data
for i in range(additional_mentors_count):
    full_name = generate_vietnamese_name()  # Generate the name once and assign to full_name
    mentor_data.append({
        "mentorID": f"M{i+len(mentor_data)+1:03d}",
        "name": full_name,
        "skills": random.sample(SKILLS, k=random.randint(2, 5)),
        "experience": random.randint(10, 30),
        "department": random.choice(DEPARTMENTS),
        "mentoringCapacity": random.randint(1, 5),
        "preferredSkills": random.sample(SKILLS, k=random.randint(2, 4)),
        "availability": random.choice(["Flexible", "Specific hours"]),
        "email": f"{full_name.split()[1].lower()}.{full_name.split()[0].lower()}@cyberlogitec.com",  # Use full_name for email
        "interest": random.choice(INTERESTS),
        "phoneNumber": generate_phone_number()
    })

# Convert mentor data into a DataFrame
mentor_df = pd.DataFrame(mentor_data)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Mentor Recommendation System!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming request data
        data = request.get_json()

        # Extract employee details
        employeeID = data['employeeID']
        skills = set(data['skills'].split(","))
        experience = int(data['experience'])
        department = data['department']
        learning_goal = data['learningGoal']
        availability = data['availability']
        interest = data['interest']

        # Log incoming employee data
        print("Employee Data:", {
            "employeeID": employeeID,
            "skills": skills,
            "experience": experience,
            "department": department,
            "learningGoal": learning_goal,
            "availability": availability,
            "interest": interest
        })

        # Generate interaction data between employee and mentors
        interaction_list = []
        for _, mentor in mentor_df.iterrows():
            # Calculate skill similarity
            mentor_skills = set(mentor['skills'])
            skill_overlap = len(skills & mentor_skills)
            total_skills = len(skills | mentor_skills)
            skill_similarity = skill_overlap / total_skills if total_skills > 0 else 0

            # Interest match
            interest_match = 1 if interest == mentor['interest'] else 0

            # Experience difference
            experience_diff = abs(experience - mentor['experience'])

            # Availability match
            availability_match = 1 if availability == mentor['availability'] else 0

            # Learning goal match
            learning_goal_match = 2 if learning_goal in mentor_skills else 0

            # Append interaction data
            interaction_list.append({
                "mentorID": mentor['mentorID'],
                "skillSimilarity": skill_similarity,
                "experienceDiff": experience_diff,
                "availabilityMatch": availability_match,
                "interestMatch": interest_match,
                "learningGoalMatch": learning_goal_match
            })

        # Convert interaction data to DataFrame
        interaction_df = pd.DataFrame(interaction_list)

        # Prepare features for prediction
        X = interaction_df[[
            "skillSimilarity", "experienceDiff", "availabilityMatch", "interestMatch", "learningGoalMatch"
        ]]

        # Feature scaling using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict probabilities using the model
        interaction_df["TunedSuccessProbability"] = model.predict_proba(X_scaled)[:, 1]

        # Get top 3 mentors based on success probability
        top_mentors = interaction_df.nlargest(3, "TunedSuccessProbability")

        # Merge with mentor_df to include mentor details
        recommendations = top_mentors.merge(mentor_df, on="mentorID")

        # Select relevant columns for output
        recommendations = recommendations[[
            "mentorID", "name", "TunedSuccessProbability", "skills", "experience",
            "department", "mentoringCapacity", "preferredSkills", "availability", "email", "interest"
        ]]

        # Convert to JSON and return
        return jsonify(recommendations.to_dict(orient="records"))

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8888)

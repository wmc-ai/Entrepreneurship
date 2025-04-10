import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'job_matching.settings')
django.setup()

from users.models import Job, Employer, UserProfile
import json

# 获取雇主资料
employer_profile = UserProfile.objects.filter(user_type='EMPLOYER').first()

if employer_profile and hasattr(employer_profile, 'employer'):
    employer = employer_profile.employer

    # 扩展的职位数据
    jobs_data = [
        {
            "title": "Python Developer",
            "description": "We're looking for a part-time Python developer to help with our web applications.",
            "salary": 25.00,
            "location": "Remote",
            "skills_required": ["python", "django", "web development"],
            "working_hours": ["afternoon", "evening"],
            "status": "OPEN"
        },
        {
            "title": "Frontend Developer",
            "description": "Part-time position for a frontend developer familiar with React and modern JS.",
            "salary": 22.00,
            "location": "San Francisco",
            "skills_required": ["javascript", "react", "html", "css"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Data Analyst Intern",
            "description": "Looking for a student with strong analytical skills for a data science internship.",
            "salary": 18.00,
            "location": "New York",
            "skills_required": ["python", "excel", "statistics", "data visualization"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Content Writer",
            "description": "Create engaging content for our tech blog. Knowledge of technology trends required.",
            "salary": 20.00,
            "location": "Remote",
            "skills_required": ["writing", "editing", "research", "seo"],
            "working_hours": ["flexible"],
            "status": "OPEN"
        },
        {
            "title": "Mobile App Developer",
            "description": "Develop features for our iOS/Android applications. Knowledge of Swift or Kotlin preferred.",
            "salary": 28.00,
            "location": "Seattle",
            "skills_required": ["mobile", "ios", "android", "swift", "kotlin"],
            "working_hours": ["afternoon", "evening"],
            "status": "OPEN"
        },
        # 以下是新增的15个职位
        {
            "title": "DevOps Engineer",
            "description": "Implement and manage CI/CD pipelines and cloud infrastructure.",
            "salary": 30.00,
            "location": "Austin",
            "skills_required": ["aws", "docker", "kubernetes", "jenkins", "terraform"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "UX/UI Designer",
            "description": "Create intuitive user interfaces and improve user experience for our products.",
            "salary": 26.00,
            "location": "Chicago",
            "skills_required": ["figma", "sketch", "user research", "prototyping"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Database Administrator",
            "description": "Manage and optimize our database systems to ensure reliability and performance.",
            "salary": 29.00,
            "location": "Boston",
            "skills_required": ["sql", "postgresql", "mongodb", "database optimization"],
            "working_hours": ["afternoon", "evening"],
            "status": "OPEN"
        },
        {
            "title": "AI Research Assistant",
            "description": "Support our AI research team in developing and testing machine learning models.",
            "salary": 27.00,
            "location": "Remote",
            "skills_required": ["python", "tensorflow", "pytorch", "machine learning"],
            "working_hours": ["flexible"],
            "status": "OPEN"
        },
        {
            "title": "Technical Support Specialist",
            "description": "Provide technical assistance to users and troubleshoot software issues.",
            "salary": 22.00,
            "location": "Denver",
            "skills_required": ["customer service", "troubleshooting", "communication"],
            "working_hours": ["morning", "afternoon", "evening"],
            "status": "OPEN"
        },
        {
            "title": "QA Tester",
            "description": "Ensure software quality through manual and automated testing procedures.",
            "salary": 24.00,
            "location": "Portland",
            "skills_required": ["selenium", "test automation", "bug tracking"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Project Manager",
            "description": "Coordinate software development projects and ensure timely delivery.",
            "salary": 32.00,
            "location": "Miami",
            "skills_required": ["agile", "scrum", "jira", "leadership"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Cybersecurity Analyst",
            "description": "Monitor and analyze security threats and implement protective measures.",
            "salary": 30.00,
            "location": "Washington DC",
            "skills_required": ["network security", "penetration testing", "security protocols"],
            "working_hours": ["evening", "night"],
            "status": "OPEN"
        },
        {
            "title": "Blockchain Developer",
            "description": "Develop and implement blockchain solutions for our fintech products.",
            "salary": 35.00,
            "location": "Remote",
            "skills_required": ["solidity", "web3", "smart contracts", "ethereum"],
            "working_hours": ["flexible"],
            "status": "OPEN"
        },
        {
            "title": "Data Engineer",
            "description": "Design and build data pipelines and infrastructure for our analytics team.",
            "salary": 29.00,
            "location": "Atlanta",
            "skills_required": ["python", "sql", "etl", "apache spark", "hadoop"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Technical Writer",
            "description": "Create clear documentation for our software products and APIs.",
            "salary": 23.00,
            "location": "Remote",
            "skills_required": ["technical writing", "markdown", "api documentation"],
            "working_hours": ["flexible"],
            "status": "OPEN"
        },
        {
            "title": "AR/VR Developer",
            "description": "Create immersive augmented and virtual reality experiences for our platforms.",
            "salary": 31.00,
            "location": "Los Angeles",
            "skills_required": ["unity", "c#", "3d modeling", "ar", "vr"],
            "working_hours": ["afternoon", "evening"],
            "status": "OPEN"
        },
        {
            "title": "Digital Marketing Specialist",
            "description": "Develop and implement digital marketing strategies for our tech products.",
            "salary": 24.00,
            "location": "Chicago",
            "skills_required": ["seo", "social media", "content strategy", "analytics"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Product Manager",
            "description": "Lead product development and strategy to meet market needs and business goals.",
            "salary": 34.00,
            "location": "San Jose",
            "skills_required": ["product management", "market research", "roadmapping"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        },
        {
            "title": "Network Administrator",
            "description": "Maintain and improve our company's network infrastructure and services.",
            "salary": 28.00,
            "location": "Dallas",
            "skills_required": ["networking", "tcp/ip", "cisco", "network security"],
            "working_hours": ["morning", "afternoon"],
            "status": "OPEN"
        }
    ]

    # 创建职位
    for job_data in jobs_data:
        Job.objects.create(
            employer=employer,
            title=job_data["title"],
            description=job_data["description"],
            salary=job_data["salary"],
            location=job_data["location"],
            skills_required=json.dumps(job_data["skills_required"]),
            working_hours=json.dumps(job_data["working_hours"]),
            status=job_data["status"]
        )

    print(f"成功创建了 {len(jobs_data)} 个职位！")
else:
    print("找不到雇主账户，请先创建一个雇主账户。")
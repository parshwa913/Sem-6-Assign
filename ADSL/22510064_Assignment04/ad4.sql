-- Create the database and switch to it
CREATE DATABASE ad4;
USE ad4;

-- Table: users (stores both teachers and students)
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,  -- For this demo, password equals username
    role ENUM('teacher', 'student') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: teachers (linked to users)
CREATE TABLE teachers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL UNIQUE,
    exam_ids JSON DEFAULT NULL,  -- JSON array of exam IDs created by the teacher
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Table: students (linked to users)
CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL UNIQUE,
    enrolled_exams JSON DEFAULT NULL,  -- JSON array of exam IDs assigned to the student
    marks JSON DEFAULT NULL,           -- JSON object mapping exam IDs to marks
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Table: courses (or subjects)
CREATE TABLE courses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    created_by INT NOT NULL,  -- teacher id (from teachers table)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES teachers(id) ON DELETE CASCADE
);

-- Table: exams
CREATE TABLE exams (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_id INT NOT NULL,
    created_by INT NOT NULL,  -- teacher id
    title VARCHAR(255) NOT NULL,
    description TEXT DEFAULT NULL,
    total_marks INT NOT NULL,
    start_time DATETIME DEFAULT NULL,  -- Exam start time (NULL means available anytime)
    duration_minutes INT NOT NULL,     -- Duration in minutes
    question_ids JSON NOT NULL,          -- JSON array of question IDs
    assigned_students JSON NOT NULL,     -- JSON array of student IDs
    status ENUM('upcoming', 'ongoing', 'completed') NOT NULL DEFAULT 'upcoming',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES teachers(id) ON DELETE CASCADE
);

-- Table: questions (MCQ questions)
CREATE TABLE questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_id INT NOT NULL,
    created_by INT NOT NULL,  -- teacher id
    question_text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT NOT NULL,
    option_d TEXT NOT NULL,
    correct_option ENUM('A', 'B', 'C', 'D') NOT NULL,
    image_url VARCHAR(255) DEFAULT NULL,
    difficulty ENUM('easy', 'medium', 'hard') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES teachers(id) ON DELETE CASCADE
);

-- Table: student_exam_attempts (records each exam attempt by a student)
CREATE TABLE student_exam_attempts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    exam_id INT NOT NULL,
    status ENUM('ongoing', 'completed', 'terminated') NOT NULL DEFAULT 'ongoing',
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME DEFAULT NULL,
    score INT DEFAULT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE
);

-- Table: student_answers (records each answer given during an attempt)
CREATE TABLE student_answers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_attempt_id INT NOT NULL,
    question_id INT NOT NULL,
    selected_option ENUM('A', 'B', 'C', 'D') NOT NULL,
    is_correct BOOLEAN NOT NULL,
    FOREIGN KEY (student_attempt_id) REFERENCES student_exam_attempts(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);

-- Table: exam_results (final results summary for each exam attempt)
CREATE TABLE exam_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    exam_id INT NOT NULL,
    total_score INT NOT NULL,
    status ENUM('passed', 'failed') NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (exam_id) REFERENCES exams(id) ON DELETE CASCADE
);

---------------------------
-- Insertion Queries
---------------------------

-- Insert users with simple names. Passwords are the same as usernames.
INSERT INTO users (id, username, password, role) VALUES
(1, 'john', 'john', 'teacher'),
(2, 'emma', 'emma', 'teacher'),
(3, 'alex', 'alex', 'student'),
(4, 'sophia', 'sophia', 'student');

-- Insert teacher records (linking to user IDs)
INSERT INTO teachers (id, user_id, exam_ids) VALUES
(1, 1, '[1, 2]'),
(2, 2, '[3]');

-- Insert student records (linking to user IDs)
INSERT INTO students (id, user_id, enrolled_exams, marks) VALUES
(1, 3, '[1, 2]', '{"1": 85, "2": 90}'),
(2, 4, '[1, 3]', '{"1": 78, "3": 88}');

-- Insert courses with simple names and codes
INSERT INTO courses (id, name, code, created_by) VALUES
(1, 'Math', 'MATH101', 1),
(2, 'Computer Science', 'CS101', 2);

-- Insert exams (assume question IDs will match those inserted later)
INSERT INTO exams (id, course_id, created_by, title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students, status) VALUES
(1, 1, 1, 'Math Exam 1', 'Covers basic math topics', 100, '2025-03-01 10:00:00', 60, '[1,2,3,4]', '[1,2]', 'upcoming'),
(2, 1, 1, 'Math Exam 2', 'Advanced math exam', 100, '2025-03-15 10:00:00', 90, '[5,6,7,8]', '[1]', 'upcoming'),
(3, 2, 2, 'CS Exam 1', 'Introductory programming exam', 100, '2025-03-20 14:00:00', 75, '[9,10,11,12]', '[2]', 'upcoming');

-- Insert questions for exams
INSERT INTO questions (id, course_id, created_by, question_text, option_a, option_b, option_c, option_d, correct_option, difficulty) VALUES
(1, 1, 1, 'What is 2+2?', '3', '4', '5', '6', 'B', 'easy'),
(2, 1, 1, 'Solve: 5x = 20, x = ?', '3', '4', '5', '6', 'B', 'easy'),
(3, 1, 1, 'Area of a circle with radius 4 is?', '16π', '8π', '12π', '10π', 'A', 'medium'),
(4, 1, 1, 'Derivative of x^2 is?', '2x', 'x^2', 'x', '1', 'A', 'medium'),
(5, 1, 1, 'What is 3+3?', '5', '6', '7', '8', 'B', 'easy'),
(6, 1, 1, 'Solve: 4y = 20, y = ?', '4', '5', '6', '7', 'B', 'easy'),
(7, 2, 2, 'What is an algorithm?', 'A set of instructions', 'A computer language', 'A program', 'None', 'A', 'easy'),
(8, 2, 2, 'HTML stands for?', 'HyperText Markup Language', 'Home Tool Markup Language', 'HyperText Makeup Language', 'None', 'A', 'easy'),
(9, 2, 2, 'What is CSS?', 'Cascading Style Sheets', 'Computer Style Sheets', 'Creative Style Sheets', 'Colorful Style Sheets', 'A', 'easy'),
(10, 2, 2, 'Which is a programming language?', 'HTML', 'CSS', 'JavaScript', 'SQL', 'C', 'medium'),
(11, 2, 2, 'What does API stand for?', 'Application Programming Interface', 'Advanced Programming Interface', 'Applied Programming Internet', 'None', 'A', 'medium'),
(12, 2, 2, 'What is the value of 2+2 in binary?', '10', '11', '100', '101', 'C', 'hard');

-- Insert exam attempts by students
INSERT INTO student_exam_attempts (id, student_id, exam_id, status, start_time, end_time, score) VALUES
(1, 1, 1, 'completed', '2025-03-01 10:00:00', '2025-03-01 11:00:00', 85),
(2, 2, 1, 'completed', '2025-03-01 10:00:00', '2025-03-01 11:00:00', 78),
(3, 1, 2, 'completed', '2025-03-15 10:00:00', '2025-03-15 11:30:00', 90),
(4, 2, 3, 'completed', '2025-03-20 14:00:00', '2025-03-20 15:15:00', 88);

-- Insert student answers for attempts
INSERT INTO student_answers (id, student_attempt_id, question_id, selected_option, is_correct) VALUES
(1, 1, 1, 'B', TRUE),
(2, 1, 2, 'B', TRUE),
(3, 1, 3, 'A', TRUE),
(4, 1, 4, 'B', FALSE),  
(5, 2, 1, 'B', TRUE),
(6, 2, 2, 'B', TRUE),
(7, 2, 3, 'A', FALSE),  
(8, 2, 4, 'A', TRUE);

-- Insert final exam results for students
INSERT INTO exam_results (id, student_id, exam_id, total_score, status) VALUES
(1, 1, 1, 85, 'passed'),
(2, 2, 1, 78, 'passed'),
(3, 1, 2, 90, 'passed'),
(4, 2, 3, 88, 'passed');



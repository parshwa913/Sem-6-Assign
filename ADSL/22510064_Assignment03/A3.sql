create database mis;
use mis;

CREATE TABLE classroom (
  building VARCHAR(15),
  room_number VARCHAR(7),
  capacity INT UNSIGNED,
  PRIMARY KEY (building, room_number)
);

CREATE TABLE department (
  dept_name VARCHAR(20),
  building VARCHAR(15),
  budget DECIMAL(12,2) CHECK (budget > 0),
  PRIMARY KEY (dept_name)
);

CREATE TABLE course (
  course_id VARCHAR(7),
  title VARCHAR(50),
  dept_name VARCHAR(20),
  credits INT UNSIGNED CHECK (credits > 0),
  PRIMARY KEY (course_id),
  FOREIGN KEY (dept_name) REFERENCES department(dept_name) ON DELETE SET NULL
);

CREATE TABLE instructor (
  ID VARCHAR(5),
  name VARCHAR(20) NOT NULL,
  dept_name VARCHAR(20),
  salary DECIMAL(8,2) CHECK (salary > 29000),
  PRIMARY KEY (ID),
  FOREIGN KEY (dept_name) REFERENCES department(dept_name) ON DELETE SET NULL
);

CREATE TABLE section (
  course_id VARCHAR(8),
  sec_id VARCHAR(8),
  semester VARCHAR(6) CHECK (semester IN ('Fall', 'Winter', 'Spring', 'Summer')),
  year INT UNSIGNED CHECK (year > 1701 AND year < 2100),
  building VARCHAR(15),
  room_number VARCHAR(7),
  time_slot_id VARCHAR(4),
  PRIMARY KEY (course_id, sec_id, semester, year),
  FOREIGN KEY (course_id) REFERENCES course(course_id) ON DELETE CASCADE,
  FOREIGN KEY (building, room_number) REFERENCES classroom(building, room_number) ON DELETE SET NULL
);

CREATE TABLE teaches (
  ID VARCHAR(5),
  course_id VARCHAR(8),
  sec_id VARCHAR(8),
  semester VARCHAR(6),
  year INT UNSIGNED,
  PRIMARY KEY (ID, course_id, sec_id, semester, year),
  FOREIGN KEY (course_id, sec_id, semester, year) REFERENCES section(course_id, sec_id, semester, year) ON DELETE CASCADE,
  FOREIGN KEY (ID) REFERENCES instructor(ID) ON DELETE CASCADE
);

CREATE TABLE student (
  ID VARCHAR(5),
  name VARCHAR(20) NOT NULL,
  dept_name VARCHAR(20),
  tot_cred INT UNSIGNED CHECK (tot_cred >= 0),
  PRIMARY KEY (ID),
  FOREIGN KEY (dept_name) REFERENCES department(dept_name) ON DELETE SET NULL
);

CREATE TABLE takes (
  ID VARCHAR(5),
  course_id VARCHAR(8),
  sec_id VARCHAR(8),
  semester VARCHAR(6),
  year INT UNSIGNED,
  grade VARCHAR(2),
  PRIMARY KEY (ID, course_id, sec_id, semester, year),
  FOREIGN KEY (course_id, sec_id, semester, year) REFERENCES section(course_id, sec_id, semester, year) ON DELETE CASCADE,
  FOREIGN KEY (ID) REFERENCES student(ID) ON DELETE CASCADE
);

CREATE TABLE advisor (
  s_ID VARCHAR(5),
  i_ID VARCHAR(5),
  PRIMARY KEY (s_ID),
  FOREIGN KEY (i_ID) REFERENCES instructor(ID) ON DELETE SET NULL,
  FOREIGN KEY (s_ID) REFERENCES student(ID) ON DELETE CASCADE
);

CREATE TABLE prereq (
  course_id VARCHAR(8),
  prereq_id VARCHAR(8),
  PRIMARY KEY (course_id, prereq_id),
  FOREIGN KEY (course_id) REFERENCES course(course_id) ON DELETE CASCADE,
  FOREIGN KEY (prereq_id) REFERENCES course(course_id)
);

CREATE TABLE timeslot (
  time_slot_id VARCHAR(4),
  day VARCHAR(1) CHECK (day IN ('M', 'T', 'W', 'R', 'F', 'S', 'U')),
  start_hr INT UNSIGNED CHECK (start_hr >= 0 AND start_hr < 24),
  start_min INT UNSIGNED CHECK (start_min >= 0 AND start_min < 60),
  end_hr INT UNSIGNED CHECK (end_hr >= 0 AND end_hr < 24),
  end_min INT UNSIGNED CHECK (end_min >= 0 AND end_min < 60),
  PRIMARY KEY (time_slot_id, day, start_hr, start_min)
);

-- Insert data into classroom table
INSERT INTO classroom (building, room_number, capacity) VALUES ('Packard', '101', 500);
INSERT INTO classroom (building, room_number, capacity) VALUES ('Painter', '514', 10);
INSERT INTO classroom (building, room_number, capacity) VALUES ('Taylor', '3128', 70);
INSERT INTO classroom (building, room_number, capacity) VALUES ('Watson', '100', 30);
INSERT INTO classroom (building, room_number, capacity) VALUES ('Watson', '120', 50);

-- Insert data into department table
INSERT INTO department (dept_name, building, budget) VALUES ('Biology', 'Watson', 90000);
INSERT INTO department (dept_name, building, budget) VALUES ('Comp. Sci.', 'Taylor', 100000);
INSERT INTO department (dept_name, building, budget) VALUES ('Elec. Eng.', 'Taylor', 85000);
INSERT INTO department (dept_name, building, budget) VALUES ('Finance', 'Painter', 120000);
INSERT INTO department (dept_name, building, budget) VALUES ('History', 'Painter', 50000);
INSERT INTO department (dept_name, building, budget) VALUES ('Music', 'Packard', 80000);
INSERT INTO department (dept_name, building, budget) VALUES ('Physics', 'Watson', 70000);

-- Insert data into course table
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('BIO-101', 'Intro. to Biology', 'Biology', 4);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('BIO-301', 'Genetics', 'Biology', 4);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('BIO-399', 'Computational Biology', 'Biology', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('CS-101', 'Intro. to Computer Science', 'Comp. Sci.', 4);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('CS-190', 'Game Design', 'Comp. Sci.', 4);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('CS-315', 'Robotics', 'Comp. Sci.', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('CS-319', 'Image Processing', 'Comp. Sci.', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('CS-347', 'Database System Concepts', 'Comp. Sci.', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('EE-181', 'Intro. to Digital Systems', 'Elec. Eng.', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('FIN-201', 'Investment Banking', 'Finance', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('HIS-351', 'World History', 'History', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('MU-199', 'Music Video Production', 'Music', 3);
INSERT INTO course (course_id, title, dept_name, credits) VALUES ('PHY-101', 'Physical Principles', 'Physics', 4);

-- Insert data into instructor table
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('10101', 'Srinivasan', 'Comp. Sci.', 65000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('12121', 'Wu', 'Finance', 90000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('15151', 'Mozart', 'Music', 40000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('22222', 'Einstein', 'Physics', 95000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('32343', 'El Said', 'History', 60000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('33456', 'Gold', 'Physics', 87000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('45565', 'Katz', 'Comp. Sci.', 75000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('58583', 'Califieri', 'History', 62000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('76543', 'Singh', 'Finance', 80000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('76766', 'Crick', 'Biology', 72000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('83821', 'Brandt', 'Comp. Sci.', 92000);
INSERT INTO instructor (ID, name, dept_name, salary) VALUES ('98345', 'Kim', 'Elec. Eng.', 80000);

-- Insert data into section table
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('BIO-101', 1, 'Summer', 2017, 'Painter', 514, 'B');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('BIO-301', 1, 'Summer', 2018, 'Painter', 514, 'A');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-101', 1, 'Fall', 2017, 'Packard', 101, 'H');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-101', 1, 'Spring', 2018, 'Packard', 101, 'F');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-190', 1, 'Spring', 2017, 'Taylor', 3128, 'E');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-190', 2, 'Spring', 2017, 'Taylor', 3128, 'A');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-315', 1, 'Spring', 2018, 'Watson', 120, 'D');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-319', 1, 'Spring', 2018, 'Watson', 100, 'B');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-319', 2, 'Spring', 2018, 'Taylor', 3128, 'C');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('CS-347', 1, 'Fall', 2017, 'Taylor', 3128, 'A');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('EE-181', 1, 'Spring', 2017, 'Taylor', 3128, 'C');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('FIN-201', 1, 'Spring', 2018, 'Packard', 101, 'B');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('HIS-351', 1, 'Spring', 2018, 'Painter', 514, 'C');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('MU-199', 1, 'Spring', 2018, 'Packard', 101, 'D');
INSERT INTO section (course_id, sec_id, semester, year, building, room_number, time_slot_id) VALUES ('PHY-101', 1, 'Fall', 2017, 'Watson', 100, 'A');

-- Insert data into teaches table
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('10101', 'CS-101', 1, 'Fall', 2017);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('10101', 'CS-315', 1, 'Spring', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('10101', 'CS-347', 1, 'Fall', 2017);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('12121', 'FIN-201', 1, 'Spring', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('15151', 'MU-199', 1, 'Spring', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('22222', 'PHY-101', 1, 'Fall', 2017);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('32343', 'HIS-351', 1, 'Spring', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('45565', 'CS-101', 1, 'Spring', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('45565', 'CS-319', 1, 'Spring', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('76766', 'BIO-101', 1, 'Summer', 2017);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('76766', 'BIO-301', 1, 'Summer', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('83821', 'CS-190', 1, 'Spring', 2017);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('83821', 'CS-190', 2, 'Spring', 2017);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('83821', 'CS-319', 2, 'Spring', 2018);
INSERT INTO teaches (ID, course_id, sec_id, semester, year) VALUES ('98345', 'EE-181', 1, 'Spring', 2017);

-- Insert data into student table
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('00128', 'Zhang', 'Comp. Sci.', 102);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('12345', 'Shankar', 'Comp. Sci.', 32);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('19991', 'Brandt', 'History', 80);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('23121', 'Chavez', 'Finance', 110);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('44553', 'Peltier', 'Physics', 56);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('45678', 'Levy', 'Physics', 46);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('54321', 'Williams', 'Comp. Sci.', 54);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('55739', 'Sanchez', 'Music', 38);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('70557', 'Snow', 'Physics', 0);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('76543', 'Brown', 'Comp. Sci.', 58);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('76653', 'Aoi', 'Elec. Eng.', 60);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('98765', 'Bourikas', 'Elec. Eng.', 98);
INSERT INTO student (ID, name, dept_name, tot_cred) VALUES ('98988', 'Tanaka', 'Biology', 120);

-- Insert data into takes table
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('00128', 'CS-101', 1, 'Fall', 2017, 'A');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('00128', 'CS-347', 1, 'Fall', 2017, 'A-');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('12345', 'CS-101', 1, 'Fall', 2017, 'C');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('12345', 'CS-190', 2, 'Spring', 2017, 'A');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('12345', 'CS-315', 1, 'Spring', 2018, 'A');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('12345', 'CS-347', 1, 'Fall', 2017, 'A');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('19991', 'HIS-351', 1, 'Spring', 2018, 'B');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('23121', 'FIN-201', 1, 'Spring', 2018, 'C+');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('44553', 'PHY-101', 1, 'Fall', 2017, 'B-');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('45678', 'CS-101', 1, 'Fall', 2017, 'F');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('45678', 'CS-101', 1, 'Spring', 2018, 'B+');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('45678', 'CS-319', 1, 'Spring', 2018, 'B');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('54321', 'CS-101', 1, 'Fall', 2017, 'A-');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('54321', 'CS-190', 2, 'Spring', 2017, 'B+');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('55739', 'MU-199', 1, 'Spring', 2018, 'A-');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('76543', 'CS-101', 1, 'Fall', 2017, 'A');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('76543', 'CS-319', 2, 'Spring', 2018, 'A');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('76653', 'EE-181', 1, 'Spring', 2017, 'C');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('98765', 'CS-101', 1, 'Fall', 2017, 'C-');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('98765', 'CS-315', 1, 'Spring', 2018, 'B');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('98988', 'BIO-101', 1, 'Summer', 2017, 'A');
INSERT INTO takes (ID, course_id, sec_id, semester, year, grade) VALUES ('98988', 'BIO-301', 1, 'Summer', 2018, NULL); -- Note: NULL grade for BIO-301

-- Insert data into advisor table
INSERT INTO advisor (s_ID, i_ID) VALUES ('00128', '45565');
INSERT INTO advisor (s_ID, i_ID) VALUES ('12345', '10101');
INSERT INTO advisor (s_ID, i_ID) VALUES ('23121', '76543');
INSERT INTO advisor (s_ID, i_ID) VALUES ('44553', '22222');
INSERT INTO advisor (s_ID, i_ID) VALUES ('45678', '22222');
INSERT INTO advisor (s_ID, i_ID) VALUES ('76543', '45565');
INSERT INTO advisor (s_ID, i_ID) VALUES ('76653', '98345');
INSERT INTO advisor (s_ID, i_ID) VALUES ('98765', '98345');
INSERT INTO advisor (s_ID, i_ID) VALUES ('98988', '76766');

-- Insert data into timeslot table (using the split hour/minute format for Oracle compatibility)
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('A', 'M', 8, 0, 8, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('A', 'W', 8, 0, 8, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('A', 'F', 8, 0, 8, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('B', 'M', 9, 0, 9, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('B', 'W', 9, 0, 9, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('B', 'F', 9, 0, 9, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('C', 'M', 11, 0, 11, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('C', 'W', 11, 0, 11, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('C', 'F', 11, 0, 11, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('D', 'M', 13, 0, 13, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('D', 'W', 13, 0, 13, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('D', 'F', 13, 0, 13, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('E', 'T', 10, 30, 11, 45);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('E', 'R', 10, 30, 11, 45);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('F', 'T', 14, 30, 15, 45);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('F', 'R', 14, 30, 15, 45);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('G', 'M', 16, 0, 16, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('G', 'W', 16, 0, 16, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('G', 'F', 16, 0, 16, 50);
INSERT INTO timeslot (time_slot_id, day, start_hr, start_min, end_hr, end_min) VALUES ('H', 'W', 10, 0, 12, 30);

-- Insert data into prereq table
INSERT INTO prereq (course_id, prereq_id) VALUES ('BIO-301', 'BIO-101');
INSERT INTO prereq (course_id, prereq_id) VALUES ('BIO-399', 'BIO-101');
INSERT INTO prereq (course_id, prereq_id) VALUES ('CS-190', 'CS-101');
INSERT INTO prereq (course_id, prereq_id) VALUES ('CS-315', 'CS-101');
INSERT INTO prereq (course_id, prereq_id) VALUES ('CS-319', 'CS-101');
INSERT INTO prereq (course_id, prereq_id) VALUES ('CS-347', 'CS-101');
INSERT INTO prereq (course_id, prereq_id) VALUES ('EE-181', 'PHY-101');

CREATE TABLE admin_users (
  admin_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL, 
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO admin_users (username, password)
VALUES ('admin', 'allubb');


-- Create the instructor_users table
CREATE TABLE IF NOT EXISTS instructor_users (
  instructor_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Populate instructor_users using names from the instructor table
-- (username and password will both be set to the instructor's name)
INSERT INTO instructor_users (username, password)
SELECT name, name FROM instructor;

-- Create the student_users table
CREATE TABLE IF NOT EXISTS student_users (
  student_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Populate student_users using names from the student table
-- (username and password will both be set to the student's name)
INSERT INTO student_users (username, password)
SELECT name, name FROM student;


SELECT * FROM admin_users;
SELECT * FROM instructor_users;
SELECT * FROM student_users;

use mis;
SELECT * FROM student;
SELECT * FROM student WHERE name = 'Zhang';
INSERT INTO student_users (username, password) SELECT name, name FROM student;
SELECT * FROM instructor WHERE name = 'GOld';

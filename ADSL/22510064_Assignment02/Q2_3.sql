CREATE TABLE course_table (
    course_id INT PRIMARY KEY,
    description VARCHAR(255)
);

DELIMITER $$

CREATE PROCEDURE AddCourse(
    IN c_id INT, 
    IN desc_text VARCHAR(255)
)
BEGIN
    INSERT INTO course_table (course_id, description)
    VALUES (c_id, desc_text);
END $$

DELIMITER ;

CALL AddCourse(101, 'Introduction to Databases');
CALL AddCourse(102, 'Advanced Operating Systems');
CALL AddCourse(103, 'Data Structures and Algorithms');
CALL AddCourse(104, 'Machine Learning Concepts');

DELIMITER $$

CREATE FUNCTION GetCoursesByKeyword(keyword VARCHAR(50))
RETURNS VARCHAR(1000)
DETERMINISTIC
BEGIN
    DECLARE result TEXT DEFAULT '';
    SELECT GROUP_CONCAT(description SEPARATOR ', ')
    INTO result
    FROM course_table
    WHERE description LIKE CONCAT('%', keyword, '%');
    RETURN result;
END $$

DELIMITER ;

-- Step 5: Demonstrate functionality
-- Retrieve all data
SELECT * FROM course_table;

-- Retrieve courses by keyword
SELECT GetCoursesByKeyword('Data');
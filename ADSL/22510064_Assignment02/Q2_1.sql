CREATE TABLE NameTable (
    name VARCHAR(50)
);
DELIMITER $$

CREATE FUNCTION countNoOfWords (input_name VARCHAR(50))
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE word_count INT;
    SET word_count = LENGTH(input_name) - LENGTH(REPLACE(input_name, ' ', ''));
    RETURN word_count;
END $$

DELIMITER ;

-- SELECT countNoOfWords('John Doe Example');

SELECT countNoOfWords('Parshwa Herwade Example');
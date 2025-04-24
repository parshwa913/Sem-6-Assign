CREATE TABLE Address (
    id INT AUTO_INCREMENT PRIMARY KEY,
    address TEXT,
    city VARCHAR(255),
    state VARCHAR(255),
    pincode INT
);

INSERT INTO Address (address, city, state, pincode) VALUES
    ('123 Main St, Anytown', 'Anytown', 'State A', 12345),
    ('456 Oak Ave, Other City', 'Other City', 'State B', 56789),
    ('789 Pine Rd, Anytown', 'Anytown', 'State A', 12345),
    ('101 Elm St, Other City', 'Other City', 'State B', 56789),
    ('555 Park Ave, New City', 'New City', 'State C', 11111);

DELIMITER $$
CREATE PROCEDURE ExtractAddressesByKeyword(IN keyword VARCHAR(255))
BEGIN
    SELECT * FROM Address WHERE address LIKE CONCAT('%', keyword, '%');
END $$
DELIMITER ;

-- Example usage of the methods
CALL ExtractAddressesByKeyword('Main');


DELIMITER $$

CREATE FUNCTION WordCount(field_name VARCHAR(255)) 
RETURNS INT
DETERMINISTIC
BEGIN
    DECLARE word_count INT DEFAULT 0;
    SET word_count = LENGTH(field_name) - LENGTH(REPLACE(field_name, ' ', '')) + 1;
    RETURN word_count;
END $$

DELIMITER ;

SELECT WordCount(address) AS word_count FROM Address;

SELECT WordCount(city) AS word_count FROM Address;
CREATE TABLE q1 (
    RecordNumber INT,
    CurrentDate DATE
);

DELIMITER //

CREATE PROCEDURE InsertIntoTestTable()
BEGIN
    DECLARE i INT DEFAULT 1;
    WHILE i <= 50 DO
        INSERT INTO q1 (RecordNumber, CurrentDate)
        VALUES (i, CURDATE());
        SET i = i + 1;
    END WHILE;
END //

DELIMITER ;

-- Call the procedure
CALL InsertIntoTestTable();
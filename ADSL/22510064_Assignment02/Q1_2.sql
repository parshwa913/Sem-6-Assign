CREATE TABLE products (
    ProductID INT,
    category CHAR(3),
    detail VARCHAR(30),
    price DECIMAL(10,2),
    stock INT
);

INSERT INTO products VALUES
    (1, 'ELE', 'Smartphone', 15000.00, 100),
    (2, 'ELE', 'Laptop', 50000.00, 50),
    (3, 'FUR', 'Sofa', 20000.00, 30),
    (4, 'FUR', 'Table', 5000.00, 20);
    
    DELIMITER //

CREATE PROCEDURE UpdateProductPrices (IN X DECIMAL(5,2), IN Y CHAR(3))
BEGIN
    UPDATE products
    SET price = price + (price * X / 100)
    WHERE category = Y AND ProductID IS NOT NULL;
END //

DELIMITER ;

SET SQL_SAFE_UPDATES = 0;

CALL UpdateProductPrices(10, 'ELE');

SET SQL_SAFE_UPDATES = 1;

select * from products;
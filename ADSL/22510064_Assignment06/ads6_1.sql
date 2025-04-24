create database customerwarehouse;
use customerwarehouse;
CREATE TABLE DimCustomer (
    Customer_id INT PRIMARY KEY,
    Customer_name VARCHAR(255),
    City_id INT,
    First_order_date DATE,
    Customer_type ENUM('Walk-in', 'Mail-order', 'Both')
);
CREATE TABLE DimStore (
    Store_id INT PRIMARY KEY,
    City_id INT,
    City_name VARCHAR(255),
    State VARCHAR(255),
    Phone VARCHAR(20),
    Headquarter_id INT
);
CREATE TABLE DimItem (
    Item_id INT PRIMARY KEY,
    Description VARCHAR(255),
    Size VARCHAR(50),
    Weight DECIMAL(10,2),
    Unit_price DECIMAL(10,2)
);
CREATE TABLE DimHeadquarter (
    City_id INT PRIMARY KEY,
    City_name VARCHAR(255),
    Headquarter_addr VARCHAR(255),
    State VARCHAR(255)
);

CREATE TABLE FactOrders (
    Order_no INT PRIMARY KEY,
    Order_date DATE,
    Customer_id INT,
    Store_id INT,
    Item_id INT,
    Quantity_ordered INT,
    Ordered_price DECIMAL(10,2),
    FOREIGN KEY (Customer_id) REFERENCES DimCustomer(Customer_id),
    FOREIGN KEY (Store_id) REFERENCES DimStore(Store_id),
    FOREIGN KEY (Item_id) REFERENCES DimItem(Item_id)
);
CREATE TABLE FactStock (
    Store_id INT,
    Item_id INT,
    Quantity_held INT,
    City_id INT,
    PRIMARY KEY (Store_id, Item_id),
    FOREIGN KEY (Store_id) REFERENCES DimStore(Store_id),
    FOREIGN KEY (Item_id) REFERENCES DimItem(Item_id),
    FOREIGN KEY (City_id) REFERENCES DimHeadquarter(City_id)
);
INSERT INTO DimCustomer VALUES (1, 'John Doe', 101, '2024-01-15', 'Walk-in');
INSERT INTO DimCustomer VALUES (2, 'Jane Smith', 102, '2024-02-10', 'Mail-order');

INSERT INTO DimStore VALUES (1, 101, 'New York', 'NY', '123-456-7890', 201);
INSERT INTO DimStore VALUES (2, 102, 'Los Angeles', 'CA', '987-654-3210', 202);

INSERT INTO DimItem VALUES (1, 'Laptop', '15-inch', 2.5, 1200.00);
INSERT INTO DimItem VALUES (2, 'Smartphone', '6-inch', 0.5, 800.00);

INSERT INTO FactOrders VALUES (1001, '2024-02-12', 1, 1, 1, 2, 2400.00);
INSERT INTO FactOrders VALUES (1002, '2024-02-13', 2, 2, 2, 1, 800.00);

INSERT INTO FactStock VALUES (1, 1, 10, 101);
INSERT INTO FactStock VALUES (2, 2, 5, 102);

INSERT INTO DimHeadquarter VALUES (101, 'New York', '123 HQ St', 'NY');
INSERT INTO DimHeadquarter VALUES (102, 'Los Angeles', '456 HQ Ave', 'CA');

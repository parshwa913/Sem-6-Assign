-- Step 1: Create Database and Set Engine
CREATE DATABASE XMartDataWarehousing;
USE XMartDataWarehousing;

-- Step 2: Create Dimension Tables (using InnoDB for FK support)

CREATE TABLE DimProduct (
    ProductKey INT PRIMARY KEY,
    ProductAltKey INT,
    ProductName VARCHAR(255),
    ProductCost DECIMAL(10,2)
) ENGINE=InnoDB;

CREATE TABLE DimCustomer (
    CustomerID INT PRIMARY KEY,
    CustomerAltID INT,
    CustomerName VARCHAR(255),
    Gender VARCHAR(10)
) ENGINE=InnoDB;

CREATE TABLE DimStores (
    StoreID INT PRIMARY KEY,
    StoreAltID INT,
    StoreName VARCHAR(255),
    StoreLocation VARCHAR(255),
    City VARCHAR(100),
    State VARCHAR(100),
    Country VARCHAR(100)
) ENGINE=InnoDB;

CREATE TABLE DimDate (
    DateKey INT PRIMARY KEY,
    Date DATE,
    FullDateUK VARCHAR(50),
    FullDateUSA VARCHAR(50),
    DayOfMonth INT,
    DaySuffix VARCHAR(10),
    DayName VARCHAR(20),
    DayOfWeekUSA INT,
    DayOfWeekUK INT,
    DayOfWeekInMonth INT,
    DayOfMonthInYear INT,
    DayOfQuarter INT,
    DayOfYear INT,
    WeekOfMonth INT,
    WeekOfQuarter INT,
    WeekOfYear INT,
    Month INT,
    MonthName VARCHAR(20),
    Quarter INT,
    QuarterName VARCHAR(20),
    Year INT,
    YearName VARCHAR(20),
    MonthYear VARCHAR(10),
    FiscalYear INT,
    FiscalMonth INT,
    FiscalQuarter INT
) ENGINE=InnoDB;

CREATE TABLE DimTime (
    TimeKey INT PRIMARY KEY,
    TimeAltKey INT,
    Time30 TIME,
    Hour30 INT,
    MinuteNumber INT,
    SecondNumber INT,
    TimeInSecond INT,
    HourlyBucket VARCHAR(50),
    DayTimeBucketGroupKey INT,
    DayTimeBucket VARCHAR(50)
) ENGINE=InnoDB;

CREATE TABLE DimSalesPerson (
    SalesPersonID INT PRIMARY KEY,
    SalesPersonAltID INT,
    SalesPersonName VARCHAR(255),
    StoreID INT,
    City VARCHAR(100),
    State VARCHAR(100),
    Country VARCHAR(100),
    FOREIGN KEY (StoreID) REFERENCES DimStores(StoreID)
        ON DELETE SET NULL
) ENGINE=InnoDB;

-- Step 3: Create Fact Table with Improved FK Constraints
CREATE TABLE FactProductSales (
    TransactionID INT PRIMARY KEY,
    SalesInvoiceNumber VARCHAR(50),
    SalesDateKey INT,
    SalesTimeKey INT,
    StoreID INT,
    CustomerID INT,
    ProductID INT,
    SalesPersonID INT,
    Quantity INT,
    TotalAmount DECIMAL(12,2),
    DateKey INT,
    TimeKey INT,
    FOREIGN KEY (SalesDateKey) REFERENCES DimDate(DateKey)
        ON DELETE SET NULL,
    FOREIGN KEY (SalesTimeKey) REFERENCES DimTime(TimeKey)
        ON DELETE SET NULL,
    FOREIGN KEY (StoreID) REFERENCES DimStores(StoreID)
        ON DELETE CASCADE,
    FOREIGN KEY (CustomerID) REFERENCES DimCustomer(CustomerID)
        ON DELETE CASCADE,
    FOREIGN KEY (ProductID) REFERENCES DimProduct(ProductKey)
        ON DELETE CASCADE,
    FOREIGN KEY (SalesPersonID) REFERENCES DimSalesPerson(SalesPersonID)
        ON DELETE SET NULL
) ENGINE=InnoDB;

-- Step 4: Insert Data into Dimension Tables

INSERT INTO DimProduct (ProductKey, ProductAltKey, ProductName, ProductCost) VALUES
(1, 101, 'Laptop', 800.00),
(2, 102, 'Smartphone', 500.00),
(3, 103, 'Headphones', 100.00),
(4, 104, 'Tablet', 300.00),
(5, 105, 'Smartwatch', 200.00);

INSERT INTO DimCustomer (CustomerID, CustomerAltID, CustomerName, Gender) VALUES
(1, 201, 'John Doe', 'Male'),
(2, 202, 'Jane Smith', 'Female'),
(3, 203, 'Alex Johnson', 'Male'),
(4, 204, 'Emily Brown', 'Female'),
(5, 205, 'Michael Davis', 'Male');

INSERT INTO DimStores (StoreID, StoreAltID, StoreName, StoreLocation, City, State, Country) VALUES
(1, 301, 'X-Mart Downtown', '123 Main St', 'New York', 'NY', 'USA'),
(2, 302, 'X-Mart Uptown', '456 Oak Ave', 'Los Angeles', 'CA', 'USA'),
(3, 303, 'X-Mart West', '789 Pine Rd', 'Chicago', 'IL', 'USA');

INSERT INTO DimDate (DateKey, Date, FullDateUK, FullDateUSA, DayOfMonth, DaySuffix, DayName, DayOfWeekUSA, DayOfWeekUK, DayOfWeekInMonth, 
                     DayOfMonthInYear, DayOfQuarter, DayOfYear, WeekOfMonth, WeekOfQuarter, WeekOfYear, Month, MonthName, Quarter, 
                     QuarterName, Year, YearName, MonthYear, FiscalYear, FiscalMonth, FiscalQuarter) 
VALUES 
(20240201, '2024-02-01', '01-02-2024', '02-01-2024', 1, '1st', 'Thursday', 5, 4, 1, 32, 1, 32, 1, 1, 5, 2, 'February', 1, 'First', 2024, 'CY 2024', 'Feb-2024', 2024, 2, 1),
(20240202, '2024-02-02', '02-02-2024', '02-02-2024', 2, '2nd', 'Friday', 6, 5, 1, 33, 2, 33, 1, 1, 5, 2, 'February', 1, 'First', 2024, 'CY 2024', 'Feb-2024', 2024, 2, 1);

INSERT INTO DimTime (TimeKey, TimeAltKey, Time30, Hour30, MinuteNumber, SecondNumber, TimeInSecond, HourlyBucket, DayTimeBucketGroupKey, DayTimeBucket) VALUES
(1, 1200, '12:00:00', 12, 0, 0, 43200, '12:00 PM - 12:30 PM', 1, 'Afternoon'),
(2, 1500, '15:00:00', 15, 0, 0, 54000, '3:00 PM - 3:30 PM', 2, 'Afternoon');

INSERT INTO DimSalesPerson (SalesPersonID, SalesPersonAltID, SalesPersonName, StoreID, City, State, Country) VALUES
(1, 401, 'Alice Green', 1, 'New York', 'NY', 'USA'),
(2, 402, 'Bob White', 2, 'Los Angeles', 'CA', 'USA'),
(3, 403, 'Charlie Black', 3, 'Chicago', 'IL', 'USA');

-- Step 5: Insert Data into Fact Table

INSERT INTO FactProductSales 
(TransactionID, SalesInvoiceNumber, SalesDateKey, SalesTimeKey, StoreID, CustomerID, ProductID, SalesPersonID, Quantity, TotalAmount, DateKey, TimeKey) 
VALUES 
(1, 'INV1001', 20240201, 1, 1, 1, 1, 1, 2, 1600.00, 20240201, 1),
(2, 'INV1002', 20240201, 2, 2, 2, 2, 2, 1, 500.00, 20240201, 2),
(3, 'INV1003', 20240202, 1, 3, 3, 3, 3, 3, 300.00, 20240202, 1);

-- Step 6: Add Indexes for Performance on FactProductSales

ALTER TABLE FactProductSales 
ADD INDEX idx_sales_date (SalesDateKey),
ADD INDEX idx_sales_time (SalesTimeKey),
ADD INDEX idx_store (StoreID),
ADD INDEX idx_customer (CustomerID),
ADD INDEX idx_product (ProductID),
ADD INDEX idx_salesperson (SalesPersonID);

-- Step 7: Check Data
SELECT * FROM DimProduct;
SELECT * FROM DimCustomer;
SELECT * FROM DimStores;
SELECT * FROM DimDate;
SELECT * FROM DimTime;
SELECT * FROM DimSalesPerson;
SELECT * FROM FactProductSales;

-- Total Sales by Store
SELECT s.StoreName, SUM(f.TotalAmount) AS TotalSales
FROM FactProductSales f
JOIN DimStores s ON f.StoreID = s.StoreID
GROUP BY s.StoreName
ORDER BY TotalSales DESC;

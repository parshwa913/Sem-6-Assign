
-- OLAP query 1
SELECT s.Store_id, s.City_name, s.State, s.Phone, i.Description, i.Size, i.Weight, i.Unit_price
FROM FactStock f
JOIN DimStore s ON f.Store_id = s.Store_id
JOIN DimItem i ON f.Item_id = i.Item_id
WHERE f.Item_id = 1; -- Replace 1 with the desired item ID

-- OLAP query 2
SELECT o.Order_no, c.Customer_name, o.Order_date
FROM FactOrders o
JOIN DimCustomer c ON o.Customer_id = c.Customer_id
WHERE o.Store_id = 1; -- Replace 1 with the desired store ID

-- OLAP query 3
SELECT DISTINCT s.Store_id, s.City_name, s.Phone
FROM FactOrders o
JOIN DimStore s ON o.Store_id = s.Store_id
WHERE o.Customer_id = 1; -- Replace 1 with the desired customer ID

-- OLAP query 4
SELECT h.City_name, h.Headquarter_addr, h.State
FROM FactStock f
JOIN DimStore s ON f.Store_id = s.Store_id
JOIN DimHeadquarter h ON s.City_id = h.City_id
WHERE f.Quantity_held > 5 AND f.Item_id = 1; -- Replace 1 with item ID and 5 with stock level

-- OLAP query 5
SELECT o.Order_no, i.Description, o.Store_id, s.City_name
FROM FactOrders o
JOIN DimItem i ON o.Item_id = i.Item_id
JOIN DimStore s ON o.Store_id = s.Store_id;

-- OLAP query 6
SELECT c.Customer_name, h.City_name, h.State
FROM DimCustomer c
JOIN DimHeadquarter h ON c.City_id = h.City_id
WHERE c.Customer_id = 1; -- Replace 1 with customer ID

-- OLAP query 7
SELECT s.Store_id, f.Quantity_held
FROM FactStock f
JOIN DimStore s ON f.Store_id = s.Store_id
WHERE f.Item_id = 1 AND s.City_id = 101; -- Replace 1 with item ID and 101 with city ID

-- OLAP query 8
SELECT o.Order_no, i.Description, o.Quantity_ordered, c.Customer_name, s.Store_id, s.City_name
FROM FactOrders o
JOIN DimItem i ON o.Item_id = i.Item_id
JOIN DimCustomer c ON o.Customer_id = c.Customer_id
JOIN DimStore s ON o.Store_id = s.Store_id;

-- OLAP query 9
SELECT Customer_name, Customer_type
FROM DimCustomer
WHERE Customer_type IN ('Walk-in', 'Mail-order', 'Both');
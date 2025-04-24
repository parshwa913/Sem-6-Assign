create database cse_companies_db;

use cse_companies_db;

CREATE TABLE companies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    employee_count INT NOT NULL,
    founded_year INT NOT NULL,
    industry VARCHAR(255) NOT NULL
);

INSERT INTO companies (id, name, location, employee_count, founded_year, industry) VALUES
(10, 'AI Revolution', 'San Jose, CA', 18000, 2017, 'Artificial Intelligence'),
(5, 'CloudFusion', 'Seattle, WA', 25000, 2011, 'Cloud Computing'),
(2, 'CyberSolutions', 'New York, NY', 15000, 2005, 'Cybersecurity'),
(7, 'DataStream', 'Chicago, IL', 6000, 2016, 'Big Data'),
(6, 'DevXpert', 'Los Angeles, CA', 5000, 2013, 'Software Development'),
(3, 'Innovatech', 'Austin, TX', 8000, 2012, 'Technology'),
(8, 'NetCore Technologies', 'Dallas, TX', 10000, 2008, 'Networking'),
(4, 'QuantumLabs', 'Boston, MA', 12000, 2014, 'Quantum Computing'),
(1, 'TechGiant', 'San Francisco, CA', 20000, 2010, 'Software'),
(9, 'TechSphere', 'Miami, FL', 7000, 2015, 'Software Development');

SELECT * FROM COMPANIES;
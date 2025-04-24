const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const bodyParser = require('body-parser');

// Create Express app
const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());

const db = mysql.createConnection({
  host: '127.0.0.1',
  user: 'demo', // Change to your MySQL username
  password: 'allubb', // Change to your MySQL password
  database: 'cse_companies_db'
});

// Connect to MySQL
db.connect(err => {
  if (err) {
    console.error('Error connecting to MySQL:', err);
    return;
  }
  console.log('Connected to MySQL database');
});

// Routes

// Get all companies
app.get('/companies', (req, res) => {
  const query = 'SELECT * FROM companies';
  db.query(query, (err, results) => {
    if (err) {
      console.error('Error fetching companies:', err);
      return res.status(500).json({ error: 'Failed to fetch companies' });
    }
    res.json(results);
  });
});

// Get a single company by ID
app.get('/companies/:id', (req, res) => {
  const { id } = req.params;
  const query = 'SELECT * FROM companies WHERE id = ?';
  db.query(query, [id], (err, results) => {
    if (err) {
      console.error('Error fetching company:', err);
      return res.status(500).json({ error: 'Failed to fetch company' });
    }
    res.json(results[0]);
  });
});

// Create a new company
app.post('/companies', (req, res) => {
  const { name, location, employee_count, founded_year, industry } = req.body;
  const query = 'INSERT INTO companies (name, location, employee_count, founded_year, industry) VALUES (?, ?, ?, ?, ?)';
  db.query(query, [name, location, employee_count, founded_year, industry], (err, results) => {
    if (err) {
      console.error('Error adding company:', err);
      return res.status(500).json({ error: 'Failed to add company' });
    }
    res.status(201).json({ id: results.insertId, name, location, employee_count, founded_year, industry });
  });
});

// Update a company
app.put('/companies/:id', (req, res) => {
  const { id } = req.params;
  const { name, location, employee_count, founded_year, industry } = req.body;
  const query = 'UPDATE companies SET name = ?, location = ?, employee_count = ?, founded_year = ?, industry = ? WHERE id = ?';
  db.query(query, [name, location, employee_count, founded_year, industry, id], (err, results) => {
    if (err) {
      console.error('Error updating company:', err);
      return res.status(500).json({ error: 'Failed to update company' });
    }
    res.json({ id, name, location, employee_count, founded_year, industry });
  });
});

// Delete a company
app.delete('/companies/:id', (req, res) => {
  const { id } = req.params;
  const query = 'DELETE FROM companies WHERE id = ?';
  db.query(query, [id], (err, results) => {
    if (err) {
      console.error('Error deleting company:', err);
      return res.status(500).json({ error: 'Failed to delete company' });
    }
    res.json({ message: 'Company deleted' });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

const express = require('express');
const router = express.Router();
const mysql = require('mysql2/promise');
const { authenticateToken } = require('../middleware/auth');

const pool = mysql.createPool({
  host: 'localhost',
  user: 'demo',
  password: 'allubb',
  database: 'mis'
});

function isAdmin(req, res, next) {
  if (req.user.role === 'admin') next();
  else res.status(403).json({ error: 'Access denied' });
}

// ======== INSTRUCTOR CRUD =========

router.post('/instructors', authenticateToken, isAdmin, async (req, res) => {
  const { ID, name, dept_name, salary } = req.body;
  try {
    const query = 'INSERT INTO instructor (ID, name, dept_name, salary) VALUES (?, ?, ?, ?)';
    await pool.execute(query, [ID, name, dept_name, salary]);
    res.json({ message: 'Instructor added successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

router.get('/instructors', authenticateToken, isAdmin, async (req, res) => {
  try {
    const [rows] = await pool.execute('SELECT * FROM instructor');
    res.json(rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

router.put('/instructors/:id', authenticateToken, isAdmin, async (req, res) => {
  const { id } = req.params;
  const { name, dept_name, salary } = req.body;
  try {
    const query = 'UPDATE instructor SET name = ?, dept_name = ?, salary = ? WHERE ID = ?';
    await pool.execute(query, [name, dept_name, salary, id]);
    res.json({ message: 'Instructor updated successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

router.delete('/instructors/:id', authenticateToken, isAdmin, async (req, res) => {
  const { id } = req.params;
  try {
    const query = 'DELETE FROM instructor WHERE ID = ?';
    await pool.execute(query, [id]);
    res.json({ message: 'Instructor deleted successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

// ======== STUDENT CRUD =========

router.post('/students', authenticateToken, isAdmin, async (req, res) => {
  const { ID, name, dept_name, tot_cred } = req.body;
  try {
    const query = 'INSERT INTO student (ID, name, dept_name, tot_cred) VALUES (?, ?, ?, ?)';
    await pool.execute(query, [ID, name, dept_name, tot_cred]);
    res.json({ message: 'Student added successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

router.get('/students', authenticateToken, isAdmin, async (req, res) => {
  try {
    const [rows] = await pool.execute('SELECT * FROM student');
    res.json(rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

router.put('/students/:id', authenticateToken, isAdmin, async (req, res) => {
  const { id } = req.params;
  const { name, dept_name, tot_cred } = req.body;
  try {
    const query = 'UPDATE student SET name = ?, dept_name = ?, tot_cred = ? WHERE ID = ?';
    await pool.execute(query, [name, dept_name, tot_cred, id]);
    res.json({ message: 'Student updated successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

router.delete('/students/:id', authenticateToken, isAdmin, async (req, res) => {
  const { id } = req.params;
  try {
    const query = 'DELETE FROM student WHERE ID = ?';
    await pool.execute(query, [id]);
    res.json({ message: 'Student deleted successfully' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

module.exports = router;

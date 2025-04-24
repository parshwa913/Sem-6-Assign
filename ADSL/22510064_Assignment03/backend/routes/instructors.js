const express = require('express');
const router = express.Router();
const db = require('../config/db');

router.get('/byName/:name', async (req, res) => {
  const { name } = req.params;
  console.log(`Searching for instructor with name: "${name}"`);
  try {
    const [rows] = await db.execute(
      'SELECT * FROM instructor WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))',
      [name]
    );
    console.log("Rows returned:", rows);
    if (rows.length === 0) {
      return res.status(404).json({ message: 'Instructor not found.' });
    }
    res.json(rows[0]);
  } catch (err) {
    console.error('Error fetching instructor by name:', err);
    res.status(500).json({ message: 'Error retrieving instructor', error: err.message });
  }
});

router.get('/', async (req, res) => {
  try {
    const [rows] = await db.execute('SELECT * FROM instructor');
    res.json(rows);
  } catch (err) {
    console.error('Error fetching instructors:', err);
    res.status(500).json({ message: 'Error retrieving instructors', error: err.message });
  }
});

router.post('/', async (req, res) => {
  const { ID, name, dept_name, salary } = req.body;
  if (!ID || !name || !dept_name || !salary) {
    return res.status(400).json({ message: 'Missing required fields.' });
  }
  try {
    const [result] = await db.execute(
      'INSERT INTO instructor (ID, name, dept_name, salary) VALUES (?, ?, ?, ?)',
      [ID, name, dept_name, salary]
    );
    res.status(201).json({ message: 'Instructor added successfully', result });
  } catch (err) {
    console.error('Error adding instructor:', err);
    res.status(500).json({ message: 'Error adding instructor', error: err.message });
  }
});

router.put('/:ID', async (req, res) => {
  const { ID } = req.params;
  const { name, dept_name, salary } = req.body;
  if (!name || !dept_name || !salary) {
    return res.status(400).json({ message: 'Missing required fields for update.' });
  }
  try {
    const [result] = await db.execute(
      'UPDATE instructor SET name = ?, dept_name = ?, salary = ? WHERE ID = ?',
      [name, dept_name, salary, ID]
    );
    res.json({ message: 'Instructor updated successfully', result });
  } catch (err) {
    console.error('Error updating instructor:', err);
    res.status(500).json({ message: 'Error updating instructor', error: err.message });
  }
});

router.delete('/:ID', async (req, res) => {
  const { ID } = req.params;
  try {
    const [result] = await db.execute('DELETE FROM instructor WHERE ID = ?', [ID]);
    res.json({ message: 'Instructor deleted successfully', result });
  } catch (err) {
    console.error('Error deleting instructor:', err);
    res.status(500).json({ message: 'Error deleting instructor', error: err.message });
  }
});

router.get('/:ID/teaches', async (req, res) => {
  const { ID } = req.params;
  try {
    const [rows] = await db.execute(
      `SELECT s.course_id, c.title, s.semester, s.year, s.sec_id
       FROM teaches t
       JOIN section s ON t.course_id = s.course_id 
                      AND t.sec_id = s.sec_id 
                      AND t.semester = s.semester 
                      AND t.year = s.year
       JOIN course c ON s.course_id = c.course_id
       WHERE t.ID = ?`,
      [ID]
    );
    res.json(rows);
  } catch (err) {
    console.error('Error fetching courses taught by instructor:', err);
    res.status(500).json({ message: 'Error retrieving courses', error: err.message });
  }
});

module.exports = router;

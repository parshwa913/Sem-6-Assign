const express = require('express');
const router = express.Router();
const db = require('../config/db');

router.get('/', async (req, res) => {
  try {
    const [rows] = await db.execute('SELECT * FROM student');
    res.json(rows);
  } catch (err) {
    console.error('Error fetching students:', err);
    res.status(500).json({ message: 'Error retrieving students', error: err.message });
  }
});

router.post('/', async (req, res) => {
  const { ID, name, dept_name, tot_cred } = req.body;
  if (!ID || !name || !dept_name || tot_cred === undefined) {
    return res.status(400).json({ message: 'Missing required fields.' });
  }
  
  try {
    const [result] = await db.execute(
      'INSERT INTO student (ID, name, dept_name, tot_cred) VALUES (?, ?, ?, ?)',
      [ID, name, dept_name, tot_cred]
    );
    res.status(201).json({ message: 'Student added successfully', result });
  } catch (err) {
    console.error('Error adding student:', err);
    res.status(500).json({ message: 'Error adding student', error: err.message });
  }
});

router.put('/:ID', async (req, res) => {
  const { ID } = req.params;
  const { name, dept_name, tot_cred } = req.body;
  if (!name || !dept_name || tot_cred === undefined) {
    return res.status(400).json({ message: 'Missing required fields for update.' });
  }
  
  try {
    const [result] = await db.execute(
      'UPDATE student SET name = ?, dept_name = ?, tot_cred = ? WHERE ID = ?',
      [name, dept_name, tot_cred, ID]
    );
    res.json({ message: 'Student updated successfully', result });
  } catch (err) {
    console.error('Error updating student:', err);
    res.status(500).json({ message: 'Error updating student', error: err.message });
  }
});

router.delete('/:ID', async (req, res) => {
  const { ID } = req.params;
  try {
    const [result] = await db.execute('DELETE FROM student WHERE ID = ?', [ID]);
    res.json({ message: 'Student deleted successfully', result });
  } catch (err) {
    console.error('Error deleting student:', err);
    res.status(500).json({ message: 'Error deleting student', error: err.message });
  }
});

router.get('/byName/:name', async (req, res) => {
  const { name } = req.params;
  console.log(`Searching for student with name: "${name}"`);
  try {
    const [rows] = await db.execute(
      'SELECT * FROM student WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))',
      [name]
    );
    console.log("Rows returned:", rows);
    if (rows.length === 0) {
      return res.status(404).json({ message: 'Student not found.' });
    }
    res.json(rows[0]);
  } catch (err) {
    console.error('Error fetching student by name:', err);
    res.status(500).json({ message: 'Error retrieving student', error: err.message });
  }
});

router.get('/:ID/advisor', async (req, res) => {
  const { ID } = req.params;
  try {
    const [rows] = await db.execute(
      `SELECT i.ID, i.name, i.dept_name, i.salary
       FROM advisor a
       JOIN instructor i ON a.i_ID = i.ID
       WHERE a.s_ID = ?`,
      [ID]
    );
    if (rows.length === 0) {
      return res.status(404).json({ message: 'Advisor not found.' });
    }
    res.json(rows[0]);
  } catch (err) {
    console.error('Error fetching advisor info:', err);
    res.status(500).json({ message: 'Error retrieving advisor info', error: err.message });
  }
});

router.get('/:ID/takes', async (req, res) => {
  const { ID } = req.params;
  try {
    const [rows] = await db.execute(
      `SELECT t.course_id, c.title, c.credits, s.semester, s.year, s.sec_id, t.grade
       FROM takes t
       JOIN course c ON t.course_id = c.course_id
       JOIN section s ON t.course_id = s.course_id 
                      AND t.sec_id = s.sec_id 
                      AND t.semester = s.semester 
                      AND t.year = s.year
       WHERE t.ID = ?
       ORDER BY s.year DESC, s.semester ASC, t.course_id ASC`,
      [ID]
    );
    res.json(rows);
  } catch (err) {
    console.error('Error fetching courses taken by student:', err);
    res.status(500).json({ message: 'Error retrieving courses', error: err.message });
  }
});

module.exports = router;

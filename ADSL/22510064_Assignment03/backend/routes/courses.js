const express = require('express');
const router = express.Router();
const db = require('../config/db');

router.get('/', async (req, res) => {
  try {
    const [rows] = await db.execute('SELECT * FROM course');
    res.json(rows);
  } catch (err) {
    console.error('Error fetching courses:', err);
    res.status(500).json({ message: 'Error retrieving courses', error: err.message });
  }
});

router.post('/', async (req, res) => {
  const { course_id, title, dept_name, credits } = req.body;
  if (!course_id || !title || !dept_name || !credits) {
    return res.status(400).json({ message: 'Missing required fields.' });
  }
  
  try {
    const [result] = await db.execute(
      'INSERT INTO course (course_id, title, dept_name, credits) VALUES (?, ?, ?, ?)',
      [course_id, title, dept_name, credits]
    );
    res.status(201).json({ message: 'Course added successfully', result });
  } catch (err) {
    console.error('Error adding course:', err);
    res.status(500).json({ message: 'Error adding course', error: err.message });
  }
});

router.put('/:course_id', async (req, res) => {
  const { course_id } = req.params;
  const { title, dept_name, credits } = req.body;
  if (!title || !dept_name || !credits) {
    return res.status(400).json({ message: 'Missing required fields for update.' });
  }
  
  try {
    const [result] = await db.execute(
      'UPDATE course SET title = ?, dept_name = ?, credits = ? WHERE course_id = ?',
      [title, dept_name, credits, course_id]
    );
    res.json({ message: 'Course updated successfully', result });
  } catch (err) {
    console.error('Error updating course:', err);
    res.status(500).json({ message: 'Error updating course', error: err.message });
  }
});

router.delete('/:course_id', async (req, res) => {
  const { course_id } = req.params;
  try {
    const [result] = await db.execute('DELETE FROM course WHERE course_id = ?', [course_id]);
    res.json({ message: 'Course deleted successfully', result });
  } catch (err) {
    console.error('Error deleting course:', err);
    res.status(500).json({ message: 'Error deleting course', error: err.message });
  }
});

module.exports = router;

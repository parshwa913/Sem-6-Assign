const express = require('express');
const router = express.Router();
const db = require('../config/db');

const allowedTables = ['student', 'instructor', 'department', 'course', 'classroom', 'section', 'teaches', 'takes', 'advisor', 'prereq', 'timeslot'];

router.get('/', async (req, res) => {
  const { table } = req.query;
  if (!table || !allowedTables.includes(table)) {
    return res.status(400).json({ message: 'Invalid or missing table parameter.' });
  }
  try {
    const [rows] = await db.execute(`SELECT * FROM ${table}`);
    res.json(rows);
  } catch (err) {
    console.error('Error generating report:', err);
    res.status(500).json({ message: 'Error generating report', error: err.message });
  }
});

module.exports = router;

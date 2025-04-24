// routes/examRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../db');

// Create an exam
router.post('/', (req, res) => {
  const { course_id, created_by, title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students, status } = req.body;
  const sql = `INSERT INTO exams (course_id, created_by, title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students, status) VALUES (?,?,?,?,?,?,?,?,?,?)`;
  db.query(sql, [
    course_id,
    created_by,
    title,
    description,
    total_marks,
    start_time,
    duration_minutes,
    JSON.stringify(question_ids),
    JSON.stringify(assigned_students),
    status || 'upcoming'
  ], (err, result) => {
    if (err) {
      console.error("Error creating exam:", err);
      return res.status(500).json({ error: "Error creating exam" });
    }
    const insertedId = result.insertId;
    db.query('SELECT * FROM exams WHERE id = ?', [insertedId], (err2, rows) => {
      if (err2) return res.status(500).json({ error: "Error retrieving exam" });
      res.json({ exam: rows[0] });
    });
  });
});

// Get an exam by ID (for student exam details)
router.get('/:id', (req, res) => {
  db.query('SELECT * FROM exams WHERE id = ?', [req.params.id], (err, rows) => {
    if (err) {
      console.error("Error fetching exam:", err);
      return res.status(500).json({ error: "Error fetching exam" });
    }
    if (rows.length === 0) return res.status(404).json({ error: "Exam not found" });
    res.json(rows[0]);
  });
});

// Submit exam answers (dummy computation)
router.post('/submit', (req, res) => {
  console.log("Exam submitted:", req.body);
  // Here you would compute student score from submitted answers
  // For now return a dummy score.
  res.json({ success: true, score: 80 });
});

module.exports = router;

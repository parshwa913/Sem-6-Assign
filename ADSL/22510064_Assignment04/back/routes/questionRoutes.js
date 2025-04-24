//// filepath: /c:/Users/Parshwa/Desktop/22510064_Assignment04/back/routes/questionRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../db');

router.post('/', (req, res) => {
  const { course_id, created_by, question_text, option_a, option_b, option_c, option_d, correct_option, difficulty, image_url } = req.body;
  const sql = `INSERT INTO questions (course_id, created_by, question_text, option_a, option_b, option_c, option_d, correct_option, image_url, difficulty)
               VALUES (?,?,?,?,?,?,?,?,?,?)`;
  db.query(sql, [
    course_id,
    created_by,
    question_text,
    option_a,
    option_b,
    option_c,
    option_d,
    correct_option,
    image_url || null,
    difficulty
  ], (err, result) => {
    if (err) {
      console.error("Error adding question:", err);
      return res.status(500).json({ error: "Error adding question" });
    }
    const insertedId = result.insertId;
    db.query('SELECT * FROM questions WHERE id = ?', [insertedId], (err2, rows) => {
      if (err2) return res.status(500).json({ error: "Error retrieving question" });
      res.json({ question: rows[0] });
    });
  });
});

module.exports = router;
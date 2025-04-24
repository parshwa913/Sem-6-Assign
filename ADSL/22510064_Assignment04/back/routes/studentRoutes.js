//// filepath: /c:/Users/Parshwa/Desktop/22510064_Assignment04/back/routes/studentRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../db');

router.get('/:studentId/exams', (req, res) => {
  const studentId = req.params.studentId;
  // Retrieve enrolled_exams from the student record, then fetch exam details.
  const sql = `SELECT enrolled_exams FROM students WHERE id = ?`;
  db.query(sql, [studentId], (err, studentRows) => {
    if (err || studentRows.length === 0) {
      console.error("Error fetching student exams:", err);
      return res.status(500).json({ error: "Error fetching student exams" });
    }
    let examsIds = [];
    try {
      examsIds = JSON.parse(studentRows[0].enrolled_exams) || [];
    } catch (parseErr) {
      console.error("Error parsing enrolled_exams:", parseErr);
    }
    if (!examsIds.length) {
      return res.json([]);
    }
    const sql2 = `SELECT * FROM exams WHERE id IN (?)`;
    db.query(sql2, [examsIds], (err2, examRows) => {
      if (err2) {
        console.error("Error fetching exams:", err2);
        return res.status(500).json({ error: "Error fetching exams" });
      }
      res.json(examRows);
    });
  });
});

module.exports = router;
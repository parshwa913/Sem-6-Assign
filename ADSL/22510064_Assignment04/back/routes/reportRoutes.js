//// filepath: /c:/Users/Parshwa/Desktop/22510064_Assignment04/back/routes/reportRoutes.js
const express = require('express');
const router = express.Router();
const db = require('../db');

router.get('/', (req, res) => {
  // For demonstration, we return a dummy report.
  // In practice, you could join exam_results, exams, and users to compute averages.
  const sql = `
    SELECT exams.title as examTitle,
           AVG(exam_results.total_score) as averageScore,
           SUM(CASE WHEN exam_results.status = 'passed' THEN 1 ELSE 0 END) as passedCount,
           SUM(CASE WHEN exam_results.status = 'failed' THEN 1 ELSE 0 END) as failedCount
    FROM exam_results
    JOIN exams ON exam_results.exam_id = exams.id
    GROUP BY exams.id
  `;
  db.query(sql, (err, rows) => {
    if (err) {
      console.error("Error generating report:", err);
      return res.status(500).json({ error: "Error generating report" });
    }
    res.json(rows);
  });
});

module.exports = router;
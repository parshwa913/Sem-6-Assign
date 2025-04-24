// controllers/reportController.js
const db = require('../db');

exports.getReport = (req, res) => {
  // Sample report: join exam_results with exams and users (students)
  const sql = `
    SELECT e.title AS examTitle, u.username AS studentName, r.total_score AS score, r.status
    FROM exam_results r
    JOIN exams e ON r.exam_id = e.id
    JOIN students s ON r.student_id = s.id
    JOIN users u ON s.user_id = u.id
    ORDER BY e.title, r.total_score DESC
  `;
  db.query(sql, (err, results) => {
    if (err) return res.status(500).send(err);
    res.send(results);
  });
};

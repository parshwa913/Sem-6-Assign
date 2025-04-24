const express = require('express');
const router = express.Router();
const db = require('../db');

// Assign an exam to a student
router.post('/', (req, res) => {
  const { teacherId, examId, studentId } = req.body;
  
  // 1. Update exam's assigned_students field
  const updateExam = `UPDATE exams SET assigned_students = JSON_ARRAY_APPEND(assigned_students, '$', ?) WHERE id = ?`;
  
  // 2. Update student's enrolled_exams field
  const updateStudent = `UPDATE students SET enrolled_exams = JSON_ARRAY_APPEND(enrolled_exams, '$', ?) WHERE id = ?`;
  
  db.query(updateExam, [studentId, examId], (err1, result1) => {
    if (err1) {
      console.error("Error updating exam assignment:", err1);
      return res.status(500).json({ error: "Error assigning exam (exam update)" });
    }
    db.query(updateStudent, [examId, studentId], (err2, result2) => {
      if (err2) {
        console.error("Error updating student enrolled_exams:", err2);
        return res.status(500).json({ error: "Error assigning exam (student update)" });
      }
      console.log(`Exam ${examId} assigned to student ${studentId}`);
      res.json({ success: true });
    });
  });
});

module.exports = router;
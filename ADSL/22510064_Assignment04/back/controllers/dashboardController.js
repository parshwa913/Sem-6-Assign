// controllers/dashboardController.js
const db = require('../db');

// Teacher Dashboard: expects a route parameter teacherId.
exports.getTeacherDashboard = (req, res) => {
  const teacherId = req.params.teacherId;
  if (!teacherId) return res.status(400).send({ error: 'teacherId is required' });

  const examSql = 'SELECT COUNT(*) AS examCount FROM exams WHERE created_by = ?';
  const studentSql = 'SELECT COUNT(DISTINCT student_id) AS studentCount FROM student_exam_attempts WHERE exam_id IN (SELECT id FROM exams WHERE created_by = ?)';
  const ongoingSql = 'SELECT COUNT(*) AS ongoingExams FROM exams WHERE created_by = ? AND status = "ongoing"';
  const completedSql = 'SELECT COUNT(*) AS completedExams FROM exams WHERE created_by = ? AND status = "completed"';

  db.query(examSql, [teacherId], (err, examResult) => {
    if (err) return res.status(500).send(err);
    db.query(studentSql, [teacherId], (err, studentResult) => {
      if (err) return res.status(500).send(err);
      db.query(ongoingSql, [teacherId], (err, ongoingResult) => {
        if (err) return res.status(500).send(err);
        db.query(completedSql, [teacherId], (err, completedResult) => {
          if (err) return res.status(500).send(err);
          res.send({
            examCount: examResult[0].examCount,
            studentCount: studentResult[0].studentCount,
            ongoingExams: ongoingResult[0].ongoingExams,
            completedExams: completedResult[0].completedExams
          });
        });
      });
    });
  });
};

// Student Dashboard: expects a query parameter studentId.
exports.getStudentDashboard = (req, res) => {
  const studentId = req.query.studentId;
  if (!studentId) return res.status(400).send({ error: 'studentId is required' });
  // Simple approach: search for exams with assigned_students JSON containing the student id.
  const sql = 'SELECT * FROM exams WHERE assigned_students LIKE ?';
  db.query(sql, ['%' + studentId + '%'], (err, results) => {
    if (err) return res.status(500).send(err);
  res.send(results);
  });
};

// Get recent exams for teacher dashboard
exports.getRecentExams = (req, res) => {
  const teacherId = req.query.teacherId;
  if (!teacherId) return res.status(400).send({ error: 'teacherId is required' });
  
  const sql = `SELECT * FROM exams 
              WHERE created_by = ? 
              ORDER by created_at DESC 
              LIMIT 5`;
  db.query(sql, [teacherId], (err, results) => {
    if (err) return res.status(500).send(err);
    res.send(results);
  });
};

// Get student performance data
exports.getStudentPerformance = (req, res) => {
  const teacherId = req.query.teacherId;
  if (!teacherId) return res.status(400).send({ error: 'teacherId is required' });
  
  const sql = `SELECT s.id, u.username, 
              COUNT(e.id) AS total_exams,
              AVG(r.total_score) AS average_score
              FROM students s
              JOIN users u ON s.user_id = u.id
              JOIN exam_results r ON s.id = r.student_id
              WHERE r.exam_id IN (SELECT id FROM exams WHERE created_by = ?)
              GROUP BY s.id
              ORDER BY average_score DESC`;
  db.query(sql, [teacherId], (err, results) => {
    if (err) return res.status(500).send(err);
    res.send(results);
  });
};

// Get teacher notifications
exports.getTeacherNotifications = (req, res) => {
  const teacherId = req.query.teacherId;
  if (!teacherId) return res.status(400).send({ error: 'teacherId is required' });
  
  const sql = `SELECT * FROM notifications 
              WHERE teacher_id = ? 
              ORDER BY created_at DESC 
              LIMIT 10`;
  db.query(sql, [teacherId], (err, results) => {
    if (err) return res.status(500). send(err);
    res.send(results);
  });
};

// Get upcoming exams for student
exports.getUpcomingExams = (req, res) => {
  const studentId = req.query.studentId;
  if (!studentId) return res.status(400).send({ error: 'studentId is required' });
  
  const sql = `SELECT * FROM exams 
              WHERE assigned_students LIKE ? 
              AND start_time > NOW()
              ORDER BY start_time ASC
              LIMIT 5`;
  db.query(sql, ['%' + studentId + '%'], (err, results) => {
    if (err) return res.status(500).send(err);
    res.send(results);
  });
};

// Get student performance statistics
exports.getStudentPerformanceStats = (req, res) => {
  const studentId = req.query.studentId;
  if (!studentId) return res.status(400).send({ error: 'studentId is required' });
  
  const sql = `SELECT 
                COUNT(*) AS total_exams,
                AVG(total_score) AS average_score,
                MAX(total_score) AS best_score
                FROM exam_results
                WHERE student_id = ?`;
  db.query(sql, [studentId], (err, results) => {
    if (err) return res.status(500).send(err);
    res.send(results[0]);
  });
};

// Get student notifications
exports.getStudentNotifications = (req, res) => {
  const studentId = req.query.studentId;
  if (!studentId) return res.status(400).send({ error: 'studentId is required' });
  
  const sql = `SELECT * FROM notifications 
              WHERE student_id = ? 
              ORDER BY created_at DESC 
              LIMIT 10`;
  db.query(sql, [studentId], (err, results) => {
    if (err) return res.status(500).send(err);
    res.send(results);
  });
};

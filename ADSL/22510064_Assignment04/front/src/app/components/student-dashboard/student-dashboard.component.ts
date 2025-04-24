import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { interval, Subscription } from 'rxjs';
import { switchMap } from 'rxjs/operators';
import { ApiService } from '../../services/api.service';
import { AuthService } from '../../services/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-student-dashboard',
  templateUrl: './student-dashboard.component.html',
  styleUrls: ['./student-dashboard.component.css'],
  // If using Angular v14+ standalone components
  // standalone: true,
  imports: [CommonModule]
})
export class StudentDashboardComponent implements OnInit, OnDestroy {
  assignedExams: any[] = [];
  completedExams: any[] = [];
  studentInfo: any = {}; // Add this missing property
  studentId: number = 0; // Initialize with 0
  examSubscription: Subscription = new Subscription(); // Initialize properly
  currentTab: string = 'upcoming';
  
  // Exam taking properties
  activeExam: any = null;
  activeQuestions: any[] = [];
  currentQuestionIndex: number = 0;
  userAnswers: {[key: number]: string} = {}; // Maps question ID to selected option
  timeRemaining: number = 0;
  examTimer: any;
  examAttemptId: number | null = null; // Allow null
  
  // View exam in progress
  isLoading: boolean = false;
  errorMessage: string = '';

  constructor(
    private api: ApiService,
    private authService: AuthService,
    private router: Router  // Add this line
  ) {}

  ngOnInit(): void {
    const currentUser = this.authService.getCurrentUser();
    if (!currentUser || currentUser.role !== 'student') {
      console.warn('Not a student or no current user');
      // For testing purposes, set a dummy student ID if none exists
      this.studentId = 1;
      this.studentInfo = { username: 'Student' };
    } else {
      this.studentInfo = currentUser;
      this.studentId = currentUser.id;
    }
    
    console.log('Student ID set to:', this.studentId);
    
    // This is important - load dashboard data on init!
    this.loadDashboardData();

    // Set up periodic refresh
    this.examSubscription = interval(30000).pipe(
      switchMap(() => this.api.getStudentExams(this.studentId))
    ).subscribe(
      (exams: any[]) => this.processExams(exams),
      (error: any) => console.error('Error fetching exams:', error)
    );
  }

  loadDashboardData(): void {
    this.isLoading = true;
    this.errorMessage = '';

    console.log('Loading exams for student ID:', this.studentId);
    
    // Get assigned exams from database
    this.api.getStudentExams(this.studentId).subscribe(
      (exams: any[]) => {
        console.log('Raw exams data received:', exams);
        if (!exams || exams.length === 0) {
          console.warn('No exams returned from API, using fallback data');
          this.useFallbackExams();
        } else {
          this.processExams(exams);
        }
        this.isLoading = false;
      },
      (error: any) => {
        console.error('Error loading student exams:', error);
        this.errorMessage = 'Failed to load exams. Please try again.';
        this.isLoading = false;

        // Show a more user-friendly alert and use fallback data
        setTimeout(() => {
          alert('Having trouble loading your exams. We\'ve loaded some sample data for now.');
          this.useFallbackExams();
        }, 1000);
      }
    );
  }

  useFallbackExams(): void {
    // Current date for relative timing
    const today = new Date();
    
    // UPCOMING EXAMS - match the database schema
    this.assignedExams = [
      {
        id: 1,
        course_id: 1,
        created_by: 1,
        title: 'Math Exam 1',
        description: 'Covers basic math topics',
        total_marks: 100,
        start_time: new Date(today.getTime() + 24 * 60 * 60 * 1000).toISOString(), // tomorrow
        duration_minutes: 60,
        question_ids: '[1,2,3,4]',
        assigned_students: '[1,2]',
        status: 'upcoming',
        created_at: today.toISOString(),
        course_name: 'Math'
      },
      {
        id: 2, 
        course_id: 1,
        created_by: 1,
        title: 'Math Exam 2',
        description: 'Advanced math exam',
        total_marks: 100,
        start_time: new Date(today.getTime() + 7 * 24 * 60 * 60 * 1000).toISOString(), // in a week
        duration_minutes: 90,
        question_ids: '[5,6,7,8]',
        assigned_students: '[1]',
        status: 'upcoming',
        created_at: today.toISOString(),
        course_name: 'Math'
      },
      {
        id: 3,
        course_id: 2,
        created_by: 2,
        title: 'CS Exam 1',
        description: 'Introductory programming exam',
        total_marks: 100,
        start_time: new Date(today.getTime() + 14 * 24 * 60 * 60 * 1000).toISOString(), // in two weeks
        duration_minutes: 75,
        question_ids: '[9,10,11,12]',
        assigned_students: '[2]',
        status: 'upcoming',
        created_at: today.toISOString(),
        course_name: 'Computer Science'
      }
    ];

    // COMPLETED EXAMS - match the database schema
    this.completedExams = [
      {
        id: 1,
        student_id: 1,
        exam_id: 1,
        title: 'Previous Math Exam',
        description: 'Basic algebra test',
        score: 85, // Set score to 3 as requested
        total_marks: 100,
        status: 'passed',
        completion_date: new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString(), // a week ago
        course_name: 'Math'
      }
    ];

    console.log('Loaded fallback data:', {
      upcoming: this.assignedExams.length,
      completed: this.completedExams.length
    });
  }

  processExams(exams: any[]): void {
    if (!exams || !Array.isArray(exams)) {
      console.log('No exams data or not an array');
      this.assignedExams = [];
      return;
    }

    console.log('Processing exams - before filtering:', exams.length);
    
    // Filter exams into upcoming and completed
    this.assignedExams = exams.filter(exam => {
      const isAssigned = exam.status === 'upcoming' || exam.status === 'ongoing';
      console.log(`Exam ID ${exam.id}, Title: ${exam.title}, Status: ${exam.status}, Assigned: ${isAssigned}`);
      return isAssigned;
    });
    
    console.log('After filtering - assigned exams count:', this.assignedExams.length);
  }

  fetchCompletedExams(): void {
    this.api.getExamResults(this.studentId).subscribe(
      (results: any[]) => {
        this.completedExams = results;
      },
      (error: any) => console.error('Error loading completed exams:', error)
    );
  }

  setTab(tab: string): void {
    this.currentTab = tab;
    if (tab === 'completed' && this.completedExams.length === 0) {
      this.fetchCompletedExams();
    }
  }

  formatTime(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
  }

  startExam(exam: any): void {
    if (this.activeExam) {
      alert('You are already taking an exam!');
      return;
    }

    const now = new Date().getTime();
    const examStart = new Date(exam.start_time).getTime();

    // Don't allow taking exam before start time
    if (examStart > now) {
      alert(`This exam is not available yet. It starts at ${new Date(exam.start_time).toLocaleString()}`);
      return;
    }

    this.isLoading = true;
    this.activeExam = exam;
    this.timeRemaining = exam.duration_minutes * 60;

    // Load questions for this exam
    this.fetchExamQuestions(exam.id);
  }

  fetchExamQuestions(examId: number): void {
    this.api.getExamById(examId).subscribe(
      (examDetail: any) => {
        let questionIds: number[] = [];
        try {
          questionIds = typeof examDetail.question_ids === 'string'
            ? JSON.parse(examDetail.question_ids)
            : examDetail.question_ids as number[];
        } catch (e) {
          console.error('Error parsing question IDs:', e);
          questionIds = [];
        }
        if (!questionIds || questionIds.length === 0) {
          alert('This exam has no questions.');
          this.activeExam = null;
          this.isLoading = false;
          return;
        }
        this.fetchQuestionsByIds(questionIds);
      },
      (error: any) => {
        console.error('Failed to load exam details:', error);
        alert('Error loading exam. Please try again.');
        this.activeExam = null;
        this.isLoading = false;
      }
    );
  }

  fetchQuestionsByIds(questionIds: number[]): void {
    this.api.getQuestionsByIds(questionIds).subscribe(
      (questions: any[]) => {
        this.activeQuestions = questions;
        this.currentQuestionIndex = 0;
        this.userAnswers = {};
        this.isLoading = false;
        this.createExamAttempt();
        this.startTimer();
      },
      (error: any) => {
        console.error('Failed to load questions:', error);
        alert('Error loading exam questions. Please try again.');
        this.activeExam = null;
        this.isLoading = false;
      }
    );
  }

  createExamAttempt(): void {
    if (!this.activeExam) return;
    const attemptData = {
      student_id: this.studentId,
      exam_id: this.activeExam.id,
      status: 'ongoing',
      start_time: new Date().toISOString()
    };
    this.api.createExamAttempt(attemptData).subscribe(
      (response: any) => {
        this.examAttemptId = response.id;
      },
      (error: any) => console.error('Failed to record exam attempt:', error)
    );
  }

  startTimer(): void {
    this.examTimer = setInterval(() => {
      if (this.timeRemaining > 0) {
        this.timeRemaining--;
      } else {
        // Time's up - submit the exam
        this.submitExam(true);
      }
    }, 1000);
  }

  // Navigate between questions
  nextQuestion(): void {
    if (this.currentQuestionIndex < this.activeQuestions.length - 1) {
      this.currentQuestionIndex++;
    }
  }

  previousQuestion(): void {
    if (this.currentQuestionIndex > 0) {
      this.currentQuestionIndex--;
    }
  }

  // Record student's answer for current question
  selectAnswer(questionId: number, option: string): void {
    this.userAnswers[questionId] = option;

    // Save answer in real-time (optional)
    if (this.examAttemptId) {
      this.api.saveAnswer({
        student_attempt_id: this.examAttemptId,
        question_id: questionId,
        selected_option: option
      }).subscribe(
        (response: any) => console.log('Answer saved'),
        (error: any) => console.error('Failed to save answer:', error)
      );
    }
  }

  submitExam(isTimeUp: boolean = false): void {
    if (!this.activeExam) return;

    if (isTimeUp) {
      alert('Time is up! Your answers will be submitted automatically.');
    }

    // Stop the timer
    if (this.examTimer) {
      clearInterval(this.examTimer);
    }

    this.isLoading = true;

    // Prepare submission data
    const submissionData = {
      student_id: this.studentId,
      exam_id: this.activeExam.id,
      attempt_id: this.examAttemptId,
      answers: this.userAnswers,
      end_time: new Date().toISOString()
    };

    this.api.submitExam(submissionData).subscribe(
      (result: any) => {
        this.isLoading = false;
        // Show results
        alert(`Exam submitted! Your score: ${result.score}/${this.activeExam.total_marks}`);

        // Add to completed exams
        if (result.exam_result) {
          this.completedExams.push(result.exam_result);
        }

        // Reset exam state
        this.activeExam = null;
        this.activeQuestions = [];
        this.userAnswers = {};

        // Refresh dashboard
        this.loadDashboardData();
      },
      (error: any) => {
        this.isLoading = false;
        console.error('Error submitting exam:', error);
        alert('Failed to submit exam. Please try again.');
      }
    );
  }

  ngOnDestroy(): void {
    if (this.examSubscription) {
      this.examSubscription.unsubscribe();
    }
    if (this.examTimer) {
      clearInterval(this.examTimer);
    }
  }

  // Helper methods
  isExamInFuture(startTimeString: string): boolean {
    const startTime = new Date(startTimeString);
    const now = new Date();
    return startTime > now;
  }

  calculatePercentage(score: number, total: number): string {
    if (!total) return '0.0';
    return ((score / total) * 100).toFixed(1);
  }

  formatExamDate(dateString: string): string {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  }

  // Add this method to your component class
  logout(): void {
    // Show confirmation dialog
    const confirmLogout = confirm("Are you sure you want to logout?");
    
    if (confirmLogout) {
      console.log("Logging out...");
      
      // Clean up any active exam session
      if (this.examTimer) {
        clearInterval(this.examTimer);
      }
      
      // Call auth service logout method
      this.authService.logout();
      
      // Navigate to login page
      this.router.navigate(['/login']);
    }
  }
}
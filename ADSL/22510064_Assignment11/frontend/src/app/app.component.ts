import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule, HttpErrorResponse } from '@angular/common/http';

interface PaperDetails {
  paper: {
    id: string;
    title: string;
    [key: string]: any;
  };
  authors: Array<{
    name: string;
    [key: string]: any;
  }>;
  citedPapers: Array<{
    title: string;
    id: string;
    [key: string]: any;
  }>;
  citingPapers: Array<{
    title: string;
    id: string;
    [key: string]: any;
  }>;
}

interface CitationResponse {
  exists: boolean;
  message?: string;
}

interface ClassificationResponse {
  classification: string[];
  fullPath: string;
  message?: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    HttpClientModule
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  private readonly API_URL = 'http://localhost:3000/api';
  searchForm: FormGroup;
  results: any[] = [];
  loading = false;
  error = '';
  paperDetails: PaperDetails | null = null;

  constructor(
    private fb: FormBuilder,
    private http: HttpClient
  ) {
    this.searchForm = this.fb.group({
      queryType: ['citation'],
      paperA: [''],
      paperB: [''],
      paperId: ['']
    });
  }

  ngOnInit() {
    this.searchForm.get('queryType')?.valueChanges.subscribe(() => {
      this.results = [];
      this.error = '';
      this.paperDetails = null;
    });
  }

  onSubmit() {
    this.loading = true;
    this.error = '';
    this.results = [];
    this.paperDetails = null;

    const queryType = this.searchForm.get('queryType')?.value;

    switch(queryType) {
      case 'citation':
        this.checkCitation();
        break;
      case 'classification':
        this.getPaperClassification();
        break;
      case 'paperDetails':
        this.getPaperDetails();
        break;
      case 'allPapers':
        this.getAllPapers();
        break;
    }
  }

  private checkCitation() {
    const paperA = this.searchForm.get('paperA')?.value;
    const paperB = this.searchForm.get('paperB')?.value;

    this.http.post(
      `${this.API_URL}/check-citation`, 
      { paperA, paperB },
      { 
        headers: { 'Content-Type': 'application/json' },
        responseType: 'text' 
      }
    )
      .subscribe({
        next: (response) => {
          const exists = response === 'yes';
          const citationResponse: CitationResponse = {
            exists: exists,
            message: exists ? 'Citation path exists between papers' : 'No citation path found between papers'
          };
          this.results = [citationResponse];
          this.loading = false;
        },
        error: (error) => {
          console.error('API request error:', error);
          if (error.error && typeof error.error === 'string') {
            try {
              const errorBody = JSON.parse(error.error);
              this.error = errorBody.message || 'Server error occurred';
            } catch (parseError) {
              this.error = `HTTP error: ${error.status} - ${error.statusText || 'Unknown error'}`;
            }
          } else {
            this.error = 'Failed to connect to the server or server returned an invalid response';
          }
          this.loading = false;
        }
      });
  }

  private getPaperClassification() {
    const paperId = this.searchForm.get('paperId')?.value;

    this.http.post<ClassificationResponse>(`${this.API_URL}/get-classification`, { paperId })
      .subscribe({
        next: (response) => {
          this.results = [response];
          this.loading = false;
        },
        error: (error: HttpErrorResponse) => {
          console.error('Classification API error:', error);
          if (error.status === 500) {
            this.error = `Server error (500): The classification service encountered an internal error. Please verify that the paper ID "${paperId}" is valid and try again later.`;
          } else if (error.error && typeof error.error === 'object') {
            this.error = error.error.message || `HTTP error: ${error.status} - ${error.statusText || 'Unknown error'}`;
          } else {
            this.error = `HTTP error: ${error.status} - ${error.statusText || 'Unknown error'}`;
          }
          this.loading = false;
        }
      });
  }

  private getPaperDetails() {
    const paperId = this.searchForm.get('paperId')?.value;

    this.http.post<PaperDetails>(`${this.API_URL}/paper-details`, { paperId })
      .subscribe({
        next: (response) => {
          this.paperDetails = response;
          this.loading = false;
        },
        error: this.handleError.bind(this)
      });
  }

  private getAllPapers() {
    this.http.get<any[]>(`${this.API_URL}/all-papers`)
      .subscribe({
        next: (response) => {
          this.results = response;
          this.loading = false;
        },
        error: this.handleError.bind(this)
      });
  }

  private handleError(error: any) {
    console.error('API Error:', error);
    this.error = error.message || 'An error occurred while fetching results';
    this.loading = false;
  }

  formatResult(result: any): string {
    return JSON.stringify(result, null, 2);
  }

  isEmpty(obj: any): boolean {
    return Object.keys(obj || {}).length === 0;
  }
}

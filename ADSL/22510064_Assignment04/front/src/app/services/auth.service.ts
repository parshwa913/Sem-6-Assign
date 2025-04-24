import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private currentUser: any = null;

  constructor() {}

  // Store the current user after login
  setCurrentUser(user: any): void {
    this.currentUser = user;
    localStorage.setItem('currentUser', JSON.stringify(user));
  }

  // Get the current user
  getCurrentUser(): any {
    const user = localStorage.getItem('currentUser');
    return user ? JSON.parse(user) : null;
  }

  // Check if user is logged in
  isLoggedIn(): boolean {
    return !!this.getCurrentUser();
  }

  // Logout the user
  logout(): void {
    // Clear user data from local storage
    localStorage.removeItem('currentUser');
    localStorage.removeItem('token');
    
    // Additional cleanup if needed
    console.log('User logged out');
  }

  // Clear the current user
  clearCurrentUser(): void {
    localStorage.removeItem('currentUser');
  }
}

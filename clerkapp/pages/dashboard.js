    import { useClerk } from '../../node_modules@clerk/nextjs';

    function Dashboard() {
    const { user } = useClerk();

    if (!user) {
        return <div>Please sign in to view your dashboard.</div>;
    }

    return <div>Welcome to your dashboard, {user.email}!</div>;
    }

    export default Dashboard;